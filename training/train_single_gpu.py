# QLoRA fine-tune of Qwen2.5-Coder-1.5B-Instruct on nvidia/OpenCodeInstruct (Python)
# Kaggle: Settings -> Accelerator: GPU T4 x2 (or P100), Internet: ON.
# Works pasted into ONE cell, or split into cells at the "# %%" markers.
# Restart the session (Run -> Restart session) before every fresh run.

# %% 1. Install (run once in its own cell, uncommented; the pip "dependency conflicts" message is harmless)
# !pip install -q -U transformers peft bitsandbytes accelerate datasets

# %% 2. Config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"      # one GPU is enough; avoids DataParallel problems
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import inspect
import torch
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID    = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATASET_ID  = "nvidia/OpenCodeInstruct"
OUT_DIR     = "/kaggle/working/qwen-coder-python-lora"
N_SAMPLES   = 10_000      # loss was already flat at ~25% of a 20k run; 10k gives most of the benefit in ~half the time
HF_USER     = "your-hf-username"                      # <-- EDIT: your Hugging Face username
BACKUP_REPO = f"{HF_USER}/qwen-coder-python-lora-backup"   # private repo that keeps your adapter safe
MAX_LEN     = 1024        # tokens per example (prompt + answer)
MIN_SCORE   = 1.0         # keep only answers that passed all their unit tests
USE_4BIT    = True        # True = QLoRA, False = plain LoRA

# --- Back up to Hugging Face so a wiped Kaggle session can never lose the work ---
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, HfApi
from transformers import TrainerCallback
login(token=UserSecretsClient().get_secret("HF_TOKEN"))
api = HfApi()
api.create_repo(BACKUP_REPO, repo_type="model", private=True, exist_ok=True)

class BackupCallback(TrainerCallback):
    """Uploads every saved checkpoint (adapter weights only, small) to the private backup repo."""
    def on_save(self, args, state, control, **kw):
        ck = f"{args.output_dir}/checkpoint-{state.global_step}"
        try:
            api.upload_folder(folder_path=ck, repo_id=BACKUP_REPO, repo_type="model",
                              path_in_repo=f"checkpoint-{state.global_step}",
                              ignore_patterns=["optimizer.pt", "scheduler.pt", "rng_state.pth"],
                              commit_message=f"checkpoint {state.global_step}")
            print("backed up", ck)
        except Exception as e:
            print("backup failed (training continues):", e)

# Short, direct system prompt: the model learns to answer with code, no hedging.
SYSTEM = "You are an expert Python programmer. Answer with correct, complete, runnable Python code."

# %% 3. Data: stream so the whole 6.4 GB is never downloaded
stream = load_dataset(DATASET_ID, split="train", streaming=True).shuffle(seed=42, buffer_size=10_000)

def good(row):
    q, a = row.get("input"), row.get("output")
    if not q or not a or "```python" not in a:
        return False
    try:
        return float(row.get("average_test_score")) >= MIN_SCORE
    except (TypeError, ValueError):
        return False

rows = []
for row in stream:
    if good(row):
        rows.append({"q": row["input"], "a": row["output"]})
        if len(rows) >= N_SAMPLES:
            break
print("kept", len(rows), "examples")
raw = Dataset.from_list(rows)

# %% 4. Tokenise with Qwen's chat template; loss only on the assistant's answer
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "right"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

def build_prompt(question):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def encode(ex):
    p_ids = tok(build_prompt(ex["q"]), add_special_tokens=False)["input_ids"]
    a_ids = tok(ex["a"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:MAX_LEN]
    labels = ([-100] * len(p_ids) + a_ids)[:MAX_LEN]
    return {"input_ids": ids, "labels": labels, "n_prompt": len(p_ids)}

ds = raw.map(encode, remove_columns=raw.column_names)
ds = ds.filter(lambda e: e["n_prompt"] < MAX_LEN - 32).remove_columns("n_prompt")
split = ds.train_test_split(test_size=200, seed=42)
train_ds, eval_ds = split["train"], split["test"]
print("train:", len(train_ds), "eval:", len(eval_ds))

def collate(batch):
    m = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id
    return {
        "input_ids":      torch.tensor([b["input_ids"] + [pad] * (m - len(b["input_ids"])) for b in batch]),
        "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (m - len(b["input_ids"])) for b in batch]),
        "labels":         torch.tensor([b["labels"] + [-100] * (m - len(b["labels"])) for b in batch]),
    }

# %% 5. Model + LoRA adapters
if USE_4BIT:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map={"": 0}, torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16))
    model = prepare_model_for_kbit_training(model)
else:
    # float32 here: fp16 mixed precision cannot train weights that are themselves fp16
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map={"": 0}, torch_dtype=torch.float32)
    model.enable_input_require_grads()

model.gradient_checkpointing_enable()
model.config.use_cache = False

model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

# %% 6. Train
wanted = dict(
    output_dir=OUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,       # effective batch 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=40,
    weight_decay=0.0,
    max_grad_norm=1.0,
    fp16=True,
    optim="paged_adamw_8bit" if USE_4BIT else "adamw_torch",
    logging_steps=20,
    save_steps=100,                      # frequent: each one is also backed up to the Hub
    save_total_limit=2,
    label_names=["labels"],
    report_to="none",
)
# Drop any argument this transformers version no longer accepts, instead of crashing.
accepted = inspect.signature(TrainingArguments.__init__).parameters
dropped = [k for k in wanted if k not in accepted]
print("dropped (unsupported) arguments:", dropped)
args = TrainingArguments(**{k: v for k, v in wanted.items() if k in accepted})

trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                  eval_dataset=eval_ds, data_collator=collate,
                  callbacks=[BackupCallback()])

has_ckpt = os.path.isdir(OUT_DIR) and any(d.startswith("checkpoint-") for d in os.listdir(OUT_DIR))
trainer.train(resume_from_checkpoint=has_ckpt)   # resumes automatically after a crash
print(trainer.evaluate())                        # note the eval_loss for the model card

model.save_pretrained(OUT_DIR + "/final")
tok.save_pretrained(OUT_DIR + "/final")
print("adapter saved to", OUT_DIR + "/final")
api.upload_folder(folder_path=OUT_DIR + "/final", repo_id=BACKUP_REPO, repo_type="model",
                  commit_message="final adapter")
print("FINAL ADAPTER BACKED UP to https://huggingface.co/" + BACKUP_REPO)

# %% 7. Try it
def ask(question, max_new_tokens=512):
    enc = tok(build_prompt(question), return_tensors="pt", add_special_tokens=False).to(model.device)
    model.eval()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

print(ask("Write a function that returns the k most frequent words in a text file."))

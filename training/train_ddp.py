# Two-GPU (DDP) QLoRA fine-tune of Qwen2.5-Coder-1.5B-Instruct on OpenCodeInstruct.
# Launched with:  torchrun --nproc_per_node=2 train_ddp.py
# Needs env vars HF_TOKEN and (optionally) HF_USER set by the launcher cell.
import os, time, inspect
import torch
from datasets import load_dataset, Dataset, load_from_disk
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, TrainerCallback)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login, HfApi

RANK       = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD      = int(os.environ.get("WORLD_SIZE", 1))
IS_MAIN    = RANK == 0
def log(*a):
    if IS_MAIN: print(*a, flush=True)

MODEL_ID    = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATASET_ID  = "nvidia/OpenCodeInstruct"
OUT_DIR     = "/kaggle/working/qwen-coder-python-lora"
DATA_DIR    = "/kaggle/working/tokenized_data"
N_SAMPLES   = 10_000
MAX_LEN     = 1024
MIN_SCORE   = 1.0
HF_USER     = os.environ["HF_USER"]
BACKUP_REPO = f"{HF_USER}/qwen-coder-python-lora-backup"
SYSTEM = "You are an expert Python programmer. Answer with correct, complete, runnable Python code."

login(token=os.environ["HF_TOKEN"])
api = HfApi()
if IS_MAIN:
    api.create_repo(BACKUP_REPO, repo_type="model", private=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "right"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

# ---- Data: rank 0 builds it once, the other process waits and loads the same copy ----
DONE = DATA_DIR + ".done"
if IS_MAIN:
    def good(row):
        q, a = row.get("input"), row.get("output")
        if not q or not a or "```python" not in a:
            return False
        try:
            return float(row.get("average_test_score")) >= MIN_SCORE
        except (TypeError, ValueError):
            return False
    stream = load_dataset(DATASET_ID, split="train", streaming=True).shuffle(seed=42, buffer_size=10_000)
    rows = []
    for row in stream:
        if good(row):
            rows.append({"q": row["input"], "a": row["output"]})
            if len(rows) >= N_SAMPLES:
                break
    print("kept", len(rows), "examples", flush=True)

    def prompt_of(q):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def encode(ex):
        p = tok(prompt_of(ex["q"]), add_special_tokens=False)["input_ids"]
        a = tok(ex["a"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        return {"input_ids": (p + a)[:MAX_LEN], "labels": ([-100] * len(p) + a)[:MAX_LEN], "n_prompt": len(p)}

    raw = Dataset.from_list(rows)
    ds = raw.map(encode, remove_columns=raw.column_names)
    ds = ds.filter(lambda e: e["n_prompt"] < MAX_LEN - 32).remove_columns("n_prompt")
    ds.train_test_split(test_size=200, seed=42).save_to_disk(DATA_DIR)
    open(DONE, "w").write("ok")
else:
    while not os.path.exists(DONE):
        time.sleep(5)
    time.sleep(3)

split = load_from_disk(DATA_DIR)
train_ds, eval_ds = split["train"], split["test"]
log("train:", len(train_ds), "eval:", len(eval_ds), "| processes:", WORLD)

def collate(batch):
    m = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id
    return {
        "input_ids":      torch.tensor([b["input_ids"] + [pad] * (m - len(b["input_ids"])) for b in batch]),
        "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (m - len(b["input_ids"])) for b in batch]),
        "labels":         torch.tensor([b["labels"] + [-100] * (m - len(b["labels"])) for b in batch]),
    }

# ---- Model: each process loads its own copy onto its own GPU ----
torch.cuda.set_device(LOCAL_RANK)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map={"": LOCAL_RANK}, torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16))
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
if IS_MAIN:
    model.print_trainable_parameters()

class BackupCallback(TrainerCallback):
    def on_save(self, args, state, control, **kw):
        if not state.is_world_process_zero:
            return
        ck = f"{args.output_dir}/checkpoint-{state.global_step}"
        try:
            api.upload_folder(folder_path=ck, repo_id=BACKUP_REPO, repo_type="model",
                              path_in_repo=f"checkpoint-{state.global_step}",
                              ignore_patterns=["optimizer.pt", "scheduler.pt", "rng_state*.pth"],
                              commit_message=f"checkpoint {state.global_step}")
            print("backed up", ck, flush=True)
        except Exception as e:
            print("backup failed (training continues):", e, flush=True)

wanted = dict(
    output_dir=OUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,          # 2 GPUs x 4 x 2 = effective batch 16 (same as before)
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=40,
    weight_decay=0.0,
    max_grad_norm=1.0,
    fp16=True,
    optim="paged_adamw_8bit",
    logging_steps=20,
    save_steps=100,
    save_total_limit=2,
    label_names=["labels"],
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
    dataloader_num_workers=0,
)
accepted = inspect.signature(TrainingArguments.__init__).parameters
log("dropped (unsupported) arguments:", [k for k in wanted if k not in accepted])
args = TrainingArguments(**{k: v for k, v in wanted.items() if k in accepted})

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                  data_collator=collate, callbacks=[BackupCallback()])
trainer.train()
metrics = trainer.evaluate()          # runs on both processes (collective), prints once
log("EVAL:", metrics)

if IS_MAIN:
    trainer.model.save_pretrained(OUT_DIR + "/final")
    tok.save_pretrained(OUT_DIR + "/final")
    print("adapter saved to", OUT_DIR + "/final", flush=True)
    api.upload_folder(folder_path=OUT_DIR + "/final", repo_id=BACKUP_REPO, repo_type="model",
                      commit_message="final adapter")
    print("FINAL ADAPTER BACKED UP to https://huggingface.co/" + BACKUP_REPO, flush=True)

# v2: shorter, gentler QLoRA run on the HARDEST examples (for the base model), on 2 GPUs.
# Launch:  torchrun --nproc_per_node=2 /kaggle/working/train_v2_ddp.py
# Env vars needed: HF_TOKEN, HF_USER.  Steps: mine hard examples (rank 0, ~20-30 min) -> train (~40 min).
import os, sys, time, inspect
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset, load_from_disk
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, TrainerCallback)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login, HfApi

def disable_torchao_check():
    for name, mod in list(sys.modules.items()):
        if name.startswith("peft") and mod is not None and hasattr(mod, "is_torchao_available"):
            mod.is_torchao_available = lambda *a, **k: False

RANK       = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD      = int(os.environ.get("WORLD_SIZE", 1))
IS_MAIN    = RANK == 0
def log(*a):
    if IS_MAIN: print(*a, flush=True)

# ------------------------------ config ------------------------------
MODEL_ID    = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATASET_ID  = "nvidia/OpenCodeInstruct"
OUT_DIR     = "/kaggle/working/qwen-coder-v2"
DATA_DIR    = "/kaggle/working/tokenized_v2"
N_POOL      = 20_000      # candidates streamed and scored by the base model
N_EVAL      = 200         # random held-out examples (NOT mined) for eval loss
N_KEEP      = 7_000       # hardest examples kept for training
DROP_TOP    = 0.03        # drop the very highest-loss 3% (often noisy or broken examples)
MAX_LEN     = 1024
LR          = 1e-4        # half of v1
MINE_BATCH  = 4
HF_USER     = os.environ["HF_USER"]
BACKUP_REPO = f"{HF_USER}/qwen-coder-python-lora-v2-backup"
SYSTEM = "You are an expert Python programmer. Answer with correct, complete, runnable Python code."

login(token=os.environ["HF_TOKEN"])
api = HfApi()
if IS_MAIN:
    api.create_repo(BACKUP_REPO, repo_type="model", private=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "right"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

def collate(batch):
    m = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id
    return {
        "input_ids":      torch.tensor([b["input_ids"] + [pad] * (m - len(b["input_ids"])) for b in batch]),
        "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (m - len(b["input_ids"])) for b in batch]),
        "labels":         torch.tensor([b["labels"] + [-100] * (m - len(b["labels"])) for b in batch]),
    }

# ------------------------------ data: rank 0 builds it, rank 1 waits ------------------------------
DONE = DATA_DIR + ".done"
if IS_MAIN:
    if os.path.exists(DONE): os.remove(DONE)

    def good(row):
        q, a = row.get("input"), row.get("output")
        if not q or not a or "```python" not in a:
            return False
        try:
            return float(row.get("average_test_score")) >= 1.0
        except (TypeError, ValueError):
            return False

    stream = load_dataset(DATASET_ID, split="train", streaming=True).shuffle(seed=7, buffer_size=10_000)
    rows = []
    for row in stream:
        if good(row):
            rows.append({"q": row["input"], "a": row["output"]})
            if len(rows) >= N_POOL:
                break
    print("pool:", len(rows), flush=True)

    def encode(ex):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ex["q"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        a = tok(ex["a"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        return {"input_ids": (p + a)[:MAX_LEN], "labels": ([-100] * len(p) + a)[:MAX_LEN], "n_prompt": len(p)}

    ds = Dataset.from_list(rows).map(encode, remove_columns=["q", "a"])
    ds = ds.filter(lambda e: e["n_prompt"] < MAX_LEN - 32).remove_columns("n_prompt")
    ds = ds.shuffle(seed=7)
    eval_ds = ds.select(range(N_EVAL))
    pool = ds.select(range(N_EVAL, len(ds)))
    print("scoring", len(pool), "candidates with the base model...", flush=True)

    # ---- mining: per-example mean loss of the BASE model on the answer tokens ----
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map={"": 0}).eval()
    lengths = [len(x) for x in pool["input_ids"]]
    order = sorted(range(len(pool)), key=lambda i: lengths[i])
    nll = [0.0] * len(pool)
    t0 = time.time()
    for bi in range(0, len(order), MINE_BATCH):
        idx = order[bi:bi + MINE_BATCH]
        b = collate([pool[i] for i in idx])
        b = {k: v.to("cuda:0") for k, v in b.items()}
        with torch.no_grad():
            logits = base(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        tgt = b["labels"][:, 1:]
        ce = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), tgt.reshape(-1),
                             ignore_index=-100, reduction="none").view(tgt.shape)
        mask = (tgt != -100).float()
        per_ex = (ce * mask).sum(1) / mask.sum(1).clamp(min=1)
        for i, v in zip(idx, per_ex.tolist()):
            nll[i] = v
        if (bi // MINE_BATCH) % 250 == 0:
            print(f"  mined {bi}/{len(order)}  ({time.time() - t0:.0f}s)", flush=True)
    del base; torch.cuda.empty_cache()

    ranked = sorted(range(len(pool)), key=lambda i: -nll[i])
    start = int(DROP_TOP * len(ranked))
    keep = ranked[start:start + N_KEEP]
    train_ds = pool.select(sorted(keep))
    mean_all = sum(nll) / len(nll)
    mean_keep = sum(nll[i] for i in keep) / len(keep)
    print(f"mean base loss: all candidates {mean_all:.3f} | kept (hard) {mean_keep:.3f} | kept {len(keep)} examples", flush=True)

    from datasets import DatasetDict
    DatasetDict({"train": train_ds, "test": eval_ds}).save_to_disk(DATA_DIR)
    open(DONE, "w").write("ok")
else:
    while not os.path.exists(DONE):
        time.sleep(10)
    time.sleep(3)

split = load_from_disk(DATA_DIR)
train_ds, eval_ds = split["train"], split["test"]
log("train:", len(train_ds), "eval:", len(eval_ds), "| processes:", WORLD)

# ------------------------------ model ------------------------------
torch.cuda.set_device(LOCAL_RANK)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map={"": LOCAL_RANK}, torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16))
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
model.config.use_cache = False
disable_torchao_check()
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
    gradient_accumulation_steps=2,          # 2 GPUs x 4 x 2 = 16
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    weight_decay=0.0,
    max_grad_norm=1.0,
    fp16=True,
    optim="paged_adamw_8bit",
    logging_steps=20,
    save_steps=100,
    save_total_limit=6,
    eval_strategy="steps",
    eval_steps=100,
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
metrics = trainer.evaluate()
log("EVAL:", metrics)

if IS_MAIN:
    trainer.model.save_pretrained(OUT_DIR + "/final")
    tok.save_pretrained(OUT_DIR + "/final")
    api.upload_folder(folder_path=OUT_DIR + "/final", repo_id=BACKUP_REPO, repo_type="model",
                      commit_message="final adapter")
    print("FINAL ADAPTER BACKED UP to https://huggingface.co/" + BACKUP_REPO, flush=True)

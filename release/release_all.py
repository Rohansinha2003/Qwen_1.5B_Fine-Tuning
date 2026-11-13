# ONE-CELL RELEASE SCRIPT: restore adapter -> merge -> test -> HumanEval -> model card -> public upload -> verify.
# Kaggle: GPU on (1 x T4 is enough), Internet on, HF_TOKEN secret attached.
# First cell (run once, separately):   !pip install -q -U transformers peft accelerate huggingface_hub datasets
# Second cell: paste this whole file and run. Takes roughly 15-25 minutes.

# ============================== CONFIG (EDIT) ==============================
HF_USER      = "2003Rohansinha"
REPO_NAME    = "Qwen2.5-Coder-1.5B-Python-OCI-LoRA"
BACKUP_REPO  = f"{HF_USER}/qwen-coder-python-lora-backup"
BASE_ID      = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
N_SAMPLES    = 10000
EPOCHS       = 1
EVAL_LOSS    = "0.133"
KNOWN_BASE_SCORE = 68.9      # HumanEval you already measured for the base model; set None to re-measure (~10 min)
RUN_EVAL     = True          # False = skip HumanEval (card then says no benchmark was run)
# ===========================================================================

import gc, os, re, subprocess, sys, tempfile, torch
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, login, snapshot_download
from kaggle_secrets import UserSecretsClient
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def disable_torchao_check():
    """Kaggle ships an old torchao that peft refuses to work with. We don't use torchao, so tell peft it is absent."""
    import sys
    for name, mod in list(sys.modules.items()):
        if name.startswith("peft") and mod is not None and hasattr(mod, "is_torchao_available"):
            mod.is_torchao_available = lambda *a, **k: False
disable_torchao_check()

REPO_ID    = f"{HF_USER}/{REPO_NAME}"
ADAPTER    = "/kaggle/working/qwen-coder-python-lora/final"
MERGED_DIR = "/kaggle/working/merged"
SYSTEM     = "You are an expert Python programmer. Answer with correct, complete, runnable Python code."

def free_gpu():
    gc.collect(); torch.cuda.empty_cache()

# ---------- 1. login ----------
TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
login(token=TOKEN)
api = HfApi(token=TOKEN)
print("1/8 logged in")

# ---------- 2. adapter ----------
if not os.path.exists(f"{ADAPTER}/adapter_model.safetensors"):
    print("adapter missing locally -> restoring from", BACKUP_REPO)
    snapshot_download(repo_id=BACKUP_REPO, token=TOKEN, local_dir=ADAPTER,
                      allow_patterns=["adapter_*", "*.json", "*.txt", "tokenizer*", "merges.txt", "vocab.json",
                                      "chat_template.jinja"],
                      ignore_patterns=["checkpoint-*"])
assert os.path.exists(f"{ADAPTER}/adapter_model.safetensors"), "adapter not found"
print("2/8 adapter ready:", sorted(os.listdir(ADAPTER)))

# ---------- 3. merge into fp16 base ----------
tok  = AutoTokenizer.from_pretrained(BASE_ID)
base = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float16, device_map={"": 0})
disable_torchao_check()
merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
merged.save_pretrained(MERGED_DIR, safe_serialization=True)
tok.save_pretrained(MERGED_DIR)
del merged, base; free_gpu()
print("3/8 merged ->", MERGED_DIR)

# ---------- 4. sanity test (aborts if the output is not code) ----------
def load_model(path):
    t = AutoTokenizer.from_pretrained(path); t.padding_side = "left"
    if t.pad_token_id is None: t.pad_token = t.eos_token
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map={"": 0}).eval()
    return m, t

SAMPLE_Q = "Write a Python function that checks whether a string is a palindrome, ignoring punctuation and case."
m, t = load_model(MERGED_DIR)
msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": SAMPLE_Q}]
enc = t(t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
        return_tensors="pt", add_special_tokens=False).to(m.device)
with torch.no_grad():
    o = m.generate(**enc, max_new_tokens=400, do_sample=False, pad_token_id=t.pad_token_id)
SAMPLE_A = t.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
del m; free_gpu()
print(SAMPLE_A)
assert "def " in SAMPLE_A, "ABORT: merged model did not produce a function. Nothing was uploaded."
print("4/8 sanity test passed")

# ---------- 5. HumanEval ----------
def extract_code(text):
    found = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.S)
    return max(found, key=len) if found else text

def build_program(p, completion):
    code = extract_code(completion)
    body = p["prompt"] + "\n" + code if f"def {p['entry_point']}" in code else p["prompt"] + code
    return body + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})\n"

def run_program(prog, timeout=15):
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "t.py"); open(f, "w").write(prog)
        try:
            return subprocess.run([sys.executable, f], capture_output=True, timeout=timeout, cwd=d).returncode == 0
        except subprocess.TimeoutExpired:
            return False

def humaneval(model_path, name, batch=8, max_new=512):
    from datasets import load_dataset
    problems = list(load_dataset("openai/openai_humaneval", split="test"))
    m, t = load_model(model_path)
    def prompt_for(p):
        user = "Complete the following Python function. Return the full function in a ```python code block.\n\n" + p["prompt"]
        return t.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                     tokenize=False, add_generation_prompt=True)
    outs = []
    for i in range(0, len(problems), batch):
        chunk = problems[i:i + batch]
        e = t([prompt_for(p) for p in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
        with torch.no_grad():
            g = m.generate(**e, max_new_tokens=max_new, do_sample=False, pad_token_id=t.pad_token_id)
        outs += t.batch_decode(g[:, e["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  {name}: generated {len(outs)}/{len(problems)}", flush=True)
    del m; free_gpu()
    with ThreadPoolExecutor(4) as ex:
        passed = list(ex.map(run_program, [build_program(p, c) for p, c in zip(problems, outs)]))
    score = 100 * sum(passed) / len(passed)
    print(f"{name}: pass@1 = {score:.1f}% ({sum(passed)}/{len(passed)})")
    return score

HUMANEVAL_BASE = HUMANEVAL_TUNED = None
if RUN_EVAL:
    HUMANEVAL_BASE = KNOWN_BASE_SCORE if KNOWN_BASE_SCORE is not None else humaneval(BASE_ID, "base")
    HUMANEVAL_TUNED = humaneval(MERGED_DIR, "fine-tuned")
    assert HUMANEVAL_TUNED > 0.5 * HUMANEVAL_BASE, (
        f"ABORT: tuned score {HUMANEVAL_TUNED:.1f} is far below base {HUMANEVAL_BASE:.1f}; merge looks broken. Nothing uploaded.")
print(f"5/8 HumanEval  base={HUMANEVAL_BASE}  tuned={HUMANEVAL_TUNED}")

# ---------- 6. model card ----------
if HUMANEVAL_BASE is not None and HUMANEVAL_TUNED is not None:
    delta = HUMANEVAL_TUNED - HUMANEVAL_BASE
    eval_section = f"""## Evaluation

HumanEval pass@1, greedy decoding, same prompt and system message for both models (164 problems):

| Model | HumanEval pass@1 |
|---|---|
| {BASE_ID} (base) | {HUMANEVAL_BASE:.1f}% |
| **{REPO_NAME}** (this model) | **{HUMANEVAL_TUNED:.1f}%** ({delta:+.1f} points) |

This is a single run with my own harness (the model's reply is parsed from a code block and executed against the
official tests), so absolute numbers may differ from other leaderboards. A difference of a few points on 164 problems
is within noise. Final validation loss on 200 held-out training-distribution examples: {EVAL_LOSS}.
"""
else:
    eval_section = f"""## Evaluation

No benchmark (HumanEval, MBPP, etc.) has been run for this model yet, so **it is not known whether it is better than
the base model** on standard benchmarks. The only metric available is the final validation loss on 200 held-out
examples from the training distribution: {EVAL_LOSS}.
"""

card = f"""---
license: apache-2.0
base_model: {BASE_ID}
datasets:
- nvidia/OpenCodeInstruct
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- code
- python
- qwen2
- lora
- qlora
- peft
---

# {REPO_NAME}

A Python code-generation model: [{BASE_ID}](https://huggingface.co/{BASE_ID}) fine-tuned with QLoRA on
{N_SAMPLES:,} test-verified Python examples from [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct).
The LoRA weights are merged into the base model, so this is a standalone model and does not need PEFT.

## Quick start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "{REPO_ID}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map="auto")

messages = [
    {{"role": "system", "content": "You are an expert Python programmer. Answer with correct, complete, runnable Python code."}},
    {{"role": "user", "content": "Write a function that merges two sorted lists."}},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**enc, max_new_tokens=512, do_sample=False)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

- Use the system message above (it was used in training). The chat template is the standard Qwen (ChatML) one.
- Needs `transformers>=4.37`. About 3 GB of memory in fp16; on CPU use `torch_dtype=torch.float32`.
- Greedy decoding (`do_sample=False`) is recommended for code.

## Example

Prompt: *{SAMPLE_Q}*

```
{SAMPLE_A}
```

## Intended use and limitations

**Intended for:** generating Python functions, classes and small scripts from a natural-language description;
learning, prototyping and research on small code models.

**Not intended for:** unreviewed production use, security-critical code, or other languages (training was Python-only).

- It is a small (1.5B) model trained on synthetic, LLM-generated data. It can produce wrong, inefficient or insecure
  code and may confidently invent APIs. **Always review and test its output.**
- Training sequences were truncated to 1,024 tokens, so very long prompts or answers may degrade.
- It inherits the biases and limitations of the base model and of the synthetic dataset.

{eval_section}
## Training details

| | |
|---|---|
| Base model | {BASE_ID} |
| Method | QLoRA: 4-bit NF4 base, LoRA r=16, alpha=32, dropout 0.05, on all attention and MLP projections (18.5M trainable parameters, 1.2%) |
| Data | {N_SAMPLES:,} Python examples from OpenCodeInstruct, keeping only answers whose generated unit tests all passed (average test score 1.0) and that contain a Python code block; max 1,024 tokens |
| Loss | On the assistant's answer only (prompt tokens masked) |
| Epochs | {EPOCHS} (about 613 steps) |
| Effective batch size | 16 (2 GPUs x 4 x 2 accumulation) |
| Optimizer / LR | paged AdamW 8-bit, 2e-4, cosine schedule, 40 warmup steps |
| Precision / hardware | fp16, 2 x NVIDIA T4 (Kaggle), about 1 hour |
| Final training loss | about 0.13 to 0.15 (mean over the run: 0.145) |
| Final validation loss | {EVAL_LOSS} |
| Frameworks | transformers, peft, bitsandbytes |

## License and attribution

- **This model:** Apache-2.0, inherited from the base model.
- **Training data:** [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) by NVIDIA, licensed CC BY 4.0.
  This model was trained on a subset of it.
- **Base model:** Qwen2.5-Coder by the Qwen team, Alibaba Cloud.

## Citation

If you use this model, please also cite the dataset and the base model:

```bibtex
@article{{ahmad2025opencodeinstruct,
  title={{OpenCodeInstruct: A Large-scale Instruction Tuning Dataset for Code LLMs}},
  author={{Ahmad, Wasi Uddin and Ficek, Aleksander and Samadi, Mehrzad and Huang, Jocelyn and Noroozi, Vahid and Majumdar, Somshubra and Ginsburg, Boris}},
  journal={{arXiv preprint arXiv:2504.04030}},
  year={{2025}}
}}

@article{{hui2024qwen2,
  title={{Qwen2.5-Coder Technical Report}},
  author={{Hui, Binyuan and others}},
  journal={{arXiv preprint arXiv:2409.12186}},
  year={{2024}}
}}
```
"""
open(f"{MERGED_DIR}/README.md", "w").write(card)
print("6/8 model card written")

# ---------- 7. upload (public) ----------
api.create_repo(REPO_ID, repo_type="model", private=False, exist_ok=True)
api.upload_folder(folder_path=MERGED_DIR, repo_id=REPO_ID, repo_type="model",
                  commit_message="Upload merged Qwen2.5-Coder-1.5B Python fine-tune",
                  ignore_patterns=["checkpoint-*", "*.pt", "*.bin"])
print("7/8 uploaded")

# ---------- 8. verify by loading it back from the Hub like a stranger ----------
m, t = load_model(REPO_ID)
enc = t(t.apply_chat_template([{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": "Write a Python fibonacci generator."}],
                              tokenize=False, add_generation_prompt=True),
        return_tensors="pt", add_special_tokens=False).to(m.device)
with torch.no_grad():
    o = m.generate(**enc, max_new_tokens=150, do_sample=False, pad_token_id=t.pad_token_id)
print(t.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
print("\n8/8 DONE. Model page: https://huggingface.co/" + REPO_ID)

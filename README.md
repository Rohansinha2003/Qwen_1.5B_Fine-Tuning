# Qwen2.5-Coder-1.5B Python fine-tune (QLoRA, free Kaggle GPUs)

Code for fine-tuning [Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
on test-verified Python examples from [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct),
evaluating it honestly against the base model, and publishing it.

**Model on Hugging Face:** [2003Rohansinha/Qwen2.5-Coder-1.5B-Python-OCI-LoRA](https://huggingface.co/2003Rohansinha/Qwen2.5-Coder-1.5B-Python-OCI-LoRA)

**Paper (draft):** [paper/One_Fine-Tune_Two_Verdicts.pdf](paper/One_Fine-Tune_Two_Verdicts.pdf)

## Results

The fine-tune is **not better than the base model on HumanEval** and scores **7.8 points higher on MBPP**.
The MBPP gain is suggestive, not proven (single run, unpaired p of about 0.07).

Pass@1 with one standard error, greedy decoding, same harness and session for all rows:

| Model | HumanEval (164) | MBPP sanitized test (257) |
|---|---|---|
| Base: Qwen2.5-Coder-1.5B-Instruct | 67.1% (±3.7), 110 solved | 59.5% (±3.1), 153 solved |
| Checkpoint step 200 | 72.0% (±3.5), 118 solved | 65.0% (±3.0), 167 solved |
| Checkpoint step 400 | 68.3% (±3.6), 112 solved | 67.3% (±2.9), 173 solved |
| **Published model (step 613)** | 65.9% (±3.7), 108 solved | 67.3% (±2.9), 173 solved |

These numbers come from `evaluation/eval_checkpoints.py` (384 new tokens, batch 16). The model card also shows
HumanEval 68.9% (base) and 67.7% (fine-tune) from `release/release_all.py`, which uses 512 new tokens and batch 8.
The two harnesses give different absolute scores, so compare models only within one table.

What the study found:

- Benchmark choice decides the conclusion: HumanEval is flat, MBPP is up.
- The gain is present by step 200 to 400. The last third of training added nothing, and the loss curve did not show that.
- Harness settings alone moved every HumanEval score by 3 problems.

## Quick use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "2003Rohansinha/Qwen2.5-Coder-1.5B-Python-OCI-LoRA"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map="auto")

messages = [
    {"role": "system", "content": "You are an expert Python programmer. Answer with correct, complete, runnable Python code."},
    {"role": "user", "content": "Write a function that merges two sorted lists."},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**enc, max_new_tokens=512, do_sample=False)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

## What is in this repository

| File | What it does |
|---|---|
| `training/train_ddp.py` | **The run behind the published model.** QLoRA on 2 GPUs (DDP), 10,000 examples, 613 steps, about 58 minutes on 2 x T4. Backs up every checkpoint to a private Hugging Face repo. |
| `training/train_single_gpu.py` | The same recipe for one GPU, as a single notebook cell. Slower (about 10 s per step on a T4). |
| `training/train_v2_hard_examples.py` | **Experimental, no results yet.** Scores 20,000 candidates with the base model and trains on the 7,000 hardest at a lower learning rate. |
| `evaluation/eval_checkpoints.py` | Scores the base model and saved checkpoints on HumanEval and MBPP, with standard errors. Produced the table above. |
| `evaluation/eval_v2.py` | Stricter evaluation for the v2 run: picks a checkpoint on validation problems, scores the test sets once, and runs a paired sign test. |
| `release/release_all.py` | Restores the adapter, merges it into the fp16 base, sanity-tests it, runs HumanEval, writes the model card, uploads the model and verifies the upload. |
| `release/update_card.py` | Adds the MBPP table to the published model card. |
| `demo/app.py` | A small Gradio app for a Hugging Face Space. |
| `paper/` | Draft paper (PDF and LaTeX source). |

## Reproduce on Kaggle

The scripts are written for Kaggle notebooks. They read your Hugging Face token from a Kaggle secret named
`HF_TOKEN` (Add-ons -> Secrets), so **no token is stored in this repository**. Turn Internet on.

**1. Install**

```python
!pip install -q -U transformers peft bitsandbytes accelerate datasets huggingface_hub
```

**2. Train (GPU T4 x2)**

Save `training/train_ddp.py` to `/kaggle/working/train_ddp.py` (for example with `%%writefile`), then:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HF_USER"] = "your-hf-username"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
!torchrun --nproc_per_node=2 /kaggle/working/train_ddp.py
```

**3. Evaluate (one GPU):** set `HF_USER` at the top of `evaluation/eval_checkpoints.py`, paste it into a cell and run it.
Set `LIMIT = 40` first for a quick smoke test.

**4. Release (one GPU):** edit the config block at the top of `release/release_all.py`, paste it into a cell and run it.

The evaluation scripts run model-generated code in subprocesses, as HumanEval and MBPP require. Run them in a
sandbox such as Kaggle, not on a machine you care about.

## Training setup

| | |
|---|---|
| Method | QLoRA: 4-bit NF4 base, LoRA rank 16, alpha 32, dropout 0.05, all attention and MLP projections |
| Trainable parameters | 18,464,768 of 1,562,179,072 (1.18%) |
| Data | 9,799 training examples: answers that passed all their generated unit tests and contain a Python code block; max 1,024 tokens |
| Schedule | 1 epoch, 613 steps, effective batch 16, learning rate 2e-4 (cosine, 40 warmup steps), fp16 |
| Loss | On the assistant answer only; final validation loss 0.133 |

## Limitations

- One training run with one seed. Scores could move by several points with another seed.
- Small benchmarks and a custom harness: the numbers are not comparable with public leaderboards.
- Library versions are not pinned. The scripts were written for the versions on Kaggle in November 2026 and
  include workarounds for removed `TrainingArguments` options and an old preinstalled `torchao`.
- The model is small and trained on synthetic data. It can write wrong or insecure code. Review and test its output.

## Licenses and credits

- Code in this repository: MIT (see `LICENSE`).
- Model weights: Apache-2.0, inherited from the base model by the Qwen team, Alibaba Cloud.
- Training data: OpenCodeInstruct by NVIDIA, CC BY 4.0. Ahmad et al., arXiv:2504.04030.
- The scripts and the paper draft were written with the help of an AI assistant (Claude, by Anthropic).

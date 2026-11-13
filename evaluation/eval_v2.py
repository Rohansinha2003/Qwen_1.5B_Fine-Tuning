# Fair test of v2 vs the base model. Paste into ONE Kaggle cell (GPU on, Internet on, HF_TOKEN attached).
# Design (so the result cannot be gamed):
#   1. Pick the best v2 checkpoint on a VALIDATION set (MBPP train+validation splits, ~163 problems) -- never on the test sets.
#   2. Score ONLY that checkpoint and the base on the TEST sets (HumanEval 164 + MBPP test 257), once.
#   3. Compare problem-by-problem (paired exact sign test), which is far more sensitive than comparing two totals.
# Roughly 1 - 1.5 hours. Results are saved to /kaggle/working/eval_v2_results.json and to the v2 backup repo.

# ============================== CONFIG (EDIT) ==============================
HF_USER     = "2003Rohansinha"
BACKUP_V2   = f"{HF_USER}/qwen-coder-python-lora-v2-backup"
BASE_ID     = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
CANDIDATES  = [100, 200, 300, 400, "final"]
BATCH       = 16
MAX_NEW     = 384          # keep identical to eval_checkpoints.py so numbers are comparable
LIMIT       = None         # e.g. 30 for a smoke test, None for the full run
# ===========================================================================
import gc, json, math, os, re, subprocess, sys, tempfile, torch
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import HfApi, snapshot_download
from kaggle_secrets import UserSecretsClient
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SYSTEM = "You are an expert Python programmer. Answer with correct, complete, runnable Python code."
OUT    = "/kaggle/working/eval_v2_results.json"
LOCAL  = "/kaggle/working/v2_adapters"
TOKEN  = UserSecretsClient().get_secret("HF_TOKEN")

def disable_torchao_check():
    for name, mod in list(sys.modules.items()):
        if name.startswith("peft") and mod is not None and hasattr(mod, "is_torchao_available"):
            mod.is_torchao_available = lambda *a, **k: False
disable_torchao_check()

def free_gpu():
    gc.collect(); torch.cuda.empty_cache()

def extract_code(text):
    found = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.S)
    return max(found, key=len) if found else text

def run_program(prog, timeout=15):
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "t.py"); open(f, "w").write(prog)
        try:
            return subprocess.run([sys.executable, f], capture_output=True, timeout=timeout, cwd=d).returncode == 0
        except subprocess.TimeoutExpired:
            return False

def humaneval_tasks():
    ds = list(load_dataset("openai/openai_humaneval", split="test"))
    out = []
    for p in ds[:LIMIT]:
        user = "Complete the following Python function. Return the full function in a ```python code block.\n\n" + p["prompt"]
        def build(c, p=p):
            code = extract_code(c)
            body = p["prompt"] + "\n" + code if f"def {p['entry_point']}" in code else p["prompt"] + code
            return body + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})\n"
        out.append((user, build))
    return out

def mbpp_tasks(splits):
    ds = list(concatenate_datasets([load_dataset("google-research-datasets/mbpp", "sanitized", split=s) for s in splits]))
    out = []
    for p in ds[:LIMIT]:
        tests = p["test_list"]
        user = (f"{p['prompt']}\nYour code should pass this test:\n{tests[0]}\n"
                "Return the full solution in a ```python code block.")
        def build(c, p=p, tests=tests):
            return "\n".join(p["test_imports"]) + "\n" + extract_code(c) + "\n\n" + "\n".join(tests) + "\n"
        out.append((user, build))
    return out

SUITES = {
    "val_mbpp":  mbpp_tasks(["train", "validation"]),   # used ONLY to choose a checkpoint
    "humaneval": humaneval_tasks(),                      # test
    "mbpp":      mbpp_tasks(["test"]),                   # test
}
print({k: len(v) for k, v in SUITES.items()})

tok = AutoTokenizer.from_pretrained(BASE_ID)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

def load_base():
    return AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float16, device_map={"": 0}).eval()

def score(model, name, suite):
    tasks = SUITES[suite]
    prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}],
                                       tokenize=False, add_generation_prompt=True) for u, _ in tasks]
    outs = []
    for i in range(0, len(prompts), BATCH):
        enc = tok(prompts[i:i + BATCH], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  {name} {suite}: {len(outs)}/{len(prompts)}", flush=True)
    with ThreadPoolExecutor(4) as ex:
        passed = list(ex.map(run_program, [b(c) for (_, b), c in zip(tasks, outs)]))
    print(f">>> {name} {suite}: {100 * sum(passed) / len(passed):.1f}% ({sum(passed)}/{len(passed)})", flush=True)
    return passed

def pct(p): return 100 * sum(p) / len(p)

def adapter_path(c):
    if c == "final":
        if not os.path.exists(f"{LOCAL}/adapter_model.safetensors"):
            snapshot_download(repo_id=BACKUP_V2, token=TOKEN, local_dir=LOCAL, allow_patterns=["adapter_*"])
        return LOCAL
    folder = f"{LOCAL}/checkpoint-{c}"
    if not os.path.exists(f"{folder}/adapter_model.safetensors"):
        snapshot_download(repo_id=BACKUP_V2, token=TOKEN, local_dir=LOCAL, allow_patterns=[f"checkpoint-{c}/adapter_*"])
    return folder

def load_merged(c):
    disable_torchao_check()
    return PeftModel.from_pretrained(load_base(), adapter_path(c)).merge_and_unload().eval()

R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save():
    json.dump(R, open(OUT, "w"))
    try:
        HfApi(token=TOKEN).upload_file(path_or_fileobj=OUT, path_in_repo="eval_v2_results.json", repo_id=BACKUP_V2,
                                       repo_type="model", commit_message="eval results")
    except Exception as e:
        print("could not back up results (continuing):", e)

# ---- pass 1: base on everything, every candidate on VALIDATION only ----
if "base" not in R:
    m = load_base()
    R["base"] = {s: score(m, "base", s) for s in SUITES}
    del m; free_gpu(); save()

for c in CANDIDATES:
    key = f"cand-{c}"
    if key in R: continue
    m = load_merged(c)
    R[key] = {"val_mbpp": score(m, key, "val_mbpp")}
    del m; free_gpu(); save()

best = None
for c in CANDIDATES:                                 # strict '>' so ties go to the earlier (less-trained) checkpoint
    v = pct(R[f"cand-{c}"]["val_mbpp"])
    if best is None or v > best[1]:
        best = (c, v)
print("\nvalidation (MBPP train+val):  base %.1f%%" % pct(R["base"]["val_mbpp"]))
for c in CANDIDATES:
    print(f"  cand-{c}: {pct(R[f'cand-{c}']['val_mbpp']):.1f}%")
print("chosen on validation:", best[0])

# ---- pass 2: test sets, chosen checkpoint only ----
ckey = f"cand-{best[0]}"
if "humaneval" not in R[ckey]:
    m = load_merged(best[0])
    for s in ("humaneval", "mbpp"):
        R[ckey][s] = score(m, ckey, s)
    del m; free_gpu(); save()

def sign_test(a, b):
    w = sum(1 for x, y in zip(a, b) if x and not y)      # chosen right, base wrong
    l = sum(1 for x, y in zip(a, b) if y and not x)      # base right, chosen wrong
    n = w + l
    if n == 0: return w, l, 1.0
    k = min(w, l)
    return w, l, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

print("\n" + "=" * 70)
print(f"TEST RESULTS  (chosen v2 checkpoint = {best[0]})")
better = []
for s, label in (("humaneval", "HumanEval"), ("mbpp", "MBPP")):
    b, c = R["base"][s], R[ckey][s]
    w, l, p = sign_test(c, b)
    print(f"{label:<10} base {pct(b):5.1f}%   v2 {pct(c):5.1f}%   diff {pct(c) - pct(b):+5.1f}   "
          f"(v2 solved {w} that base missed, base solved {l} that v2 missed, p={p:.3f})")
    better.append(pct(c) > pct(b))
if all(better):
    print("VERDICT: v2 scores higher than the base on BOTH benchmarks.")
elif any(better):
    print("VERDICT: mixed -- higher on one benchmark, not the other.")
else:
    print("VERDICT: v2 is NOT better than the base on these benchmarks.")
print("(p < 0.05 would be strong evidence; 0.05-0.15 is suggestive; above that, treat as noise.)")

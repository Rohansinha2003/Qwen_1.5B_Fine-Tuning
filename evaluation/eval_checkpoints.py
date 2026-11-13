# Score the base model and several of your saved checkpoints on HumanEval (164) + MBPP-sanitized test (257).
# Kaggle: GPU on (1 x T4 is enough), Internet on, HF_TOKEN secret attached. Paste into ONE cell and run.
# Expect roughly 1 - 1.5 hours for 4 models. Results are saved after every model so a crash loses nothing.
# NOTE: this runs model-generated code in subprocesses (standard for these benchmarks); fine inside Kaggle's sandbox.

# ============================== CONFIG (EDIT) ==============================
HF_USER     = "2003Rohansinha"
BACKUP_REPO = f"{HF_USER}/qwen-coder-python-lora-backup"
BASE_ID     = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
CHECKPOINTS = [200, 400, 613]     # 613 = the final adapter you published
INCLUDE_BASE = True
BATCH       = 16                  # lower to 8 if you hit out-of-memory
MAX_NEW     = 384
LIMIT       = None                # e.g. 40 for a quick smoke test of the whole pipeline, None for the full run
# ===========================================================================

import gc, json, math, os, re, subprocess, sys, tempfile, torch
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from kaggle_secrets import UserSecretsClient
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SYSTEM  = "You are an expert Python programmer. Answer with correct, complete, runnable Python code."
OUT     = "/kaggle/working/eval_results.json"
TOKEN   = UserSecretsClient().get_secret("HF_TOKEN")

def disable_torchao_check():
    """Kaggle ships an old torchao that peft refuses to work with; we don't use torchao."""
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

# ---------- benchmark tasks: each is (user_prompt, builder(completion) -> program) ----------
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

def mbpp_tasks():
    ds = list(load_dataset("google-research-datasets/mbpp", "sanitized", split="test"))
    out = []
    for p in ds[:LIMIT]:
        tests = p["test_list"]
        user = (f"{p['prompt']}\nYour code should pass this test:\n{tests[0]}\n"
                "Return the full solution in a ```python code block.")
        def build(c, p=p, tests=tests):
            return "\n".join(p["test_imports"]) + "\n" + extract_code(c) + "\n\n" + "\n".join(tests) + "\n"
        out.append((user, build))
    return out

SUITES = {"humaneval": humaneval_tasks(), "mbpp": mbpp_tasks()}
print({k: len(v) for k, v in SUITES.items()})

def score_suite(model, tok, name, suite):
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
    n, k = len(passed), sum(passed)
    p = k / n
    return {"pass": k, "n": n, "pct": 100 * p, "se": 100 * math.sqrt(p * (1 - p) / n)}

def load_base():
    return AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float16, device_map={"": 0}).eval()

tok = AutoTokenizer.from_pretrained(BASE_ID)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

results = json.load(open(OUT)) if os.path.exists(OUT) else {}
api = HfApi(token=TOKEN)

def save_results():
    json.dump(results, open(OUT, "w"), indent=2)
    try:
        api.upload_file(path_or_fileobj=OUT, path_in_repo="eval_results.json", repo_id=BACKUP_REPO,
                        repo_type="model", commit_message="eval results")
    except Exception as e:
        print("could not back up results (continuing):", e)

def run_one(name, make_model):
    if name in results:
        print(f"skip {name} (already scored)"); return
    model = make_model()
    results[name] = {s: score_suite(model, tok, name, s) for s in SUITES}
    del model; free_gpu()
    save_results()
    r = results[name]
    print(f">>> {name}: HumanEval {r['humaneval']['pct']:.1f}%  MBPP {r['mbpp']['pct']:.1f}%", flush=True)

if INCLUDE_BASE:
    run_one("base", load_base)

for step in CHECKPOINTS:
    folder = f"/kaggle/working/ckpt_eval/checkpoint-{step}"
    if not os.path.exists(f"{folder}/adapter_model.safetensors"):
        snapshot_download(repo_id=BACKUP_REPO, token=TOKEN, local_dir="/kaggle/working/ckpt_eval",
                          allow_patterns=[f"checkpoint-{step}/adapter_*"])
    def make(folder=folder):
        disable_torchao_check()
        return PeftModel.from_pretrained(load_base(), folder).merge_and_unload().eval()
    run_one(f"step-{step}", make)

# ---------- summary ----------
print("\n" + "=" * 64)
print(f"{'model':<12}{'HumanEval':>22}{'MBPP':>26}")
for name, r in results.items():
    h, m = r["humaneval"], r["mbpp"]
    print(f"{name:<12}{h['pct']:>12.1f}% (+-{h['se']:.1f}) {h['pass']}/{h['n']:<4}{m['pct']:>9.1f}% (+-{m['se']:.1f}) {m['pass']}/{m['n']}")
print("(+- is one standard error: differences smaller than about 2x this are noise.)")

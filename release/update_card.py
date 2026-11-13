# Adds the MBPP result (with error bars) to the published model card. Run in any Kaggle cell (Internet on, HF_TOKEN attached).
# Idempotent: running it twice does not duplicate the section.
from huggingface_hub import hf_hub_download, HfApi
from kaggle_secrets import UserSecretsClient

REPO_ID = "2003Rohansinha/Qwen2.5-Coder-1.5B-Python-OCI-LoRA"
TOKEN = UserSecretsClient().get_secret("HF_TOKEN")

path = hf_hub_download(REPO_ID, "README.md", token=TOKEN)
text = open(path, encoding="utf-8").read()

section = """### Second benchmark (MBPP) and error bars

A second evaluation run used a slightly different harness (answers capped at 384 new tokens, batch 16), so its
numbers are **not** directly comparable to the table above; compare models only within this table.
HumanEval = 164 problems, MBPP (sanitized test split) = 257 problems; the value in brackets is one standard error.

| Model | HumanEval pass@1 | MBPP pass@1 |
|---|---|---|
| Base: Qwen2.5-Coder-1.5B-Instruct | 67.1% (±3.7) | 59.5% (±3.1) |
| This model (final, step 613) | 65.9% (±3.7) | 67.3% (±2.9) |

Reading: on MBPP this model scores about 8 points higher than the base, which is suggestive but, with these error
bars, not conclusive on its own. On HumanEval there is no improvement. Intermediate checkpoints (not published) scored
within noise of each other, with most of the MBPP gain already present by step 200 of 613.

"""

if "### Second benchmark (MBPP)" in text:
    print("section already present; nothing to do")
else:
    assert "## Training details" in text, "unexpected README layout; not modified"
    text = text.replace("## Training details", section + "## Training details", 1)
    open("README_new.md", "w", encoding="utf-8").write(text)
    HfApi(token=TOKEN).upload_file(path_or_fileobj="README_new.md", path_in_repo="README.md", repo_id=REPO_ID,
                                   repo_type="model", commit_message="Add MBPP results with error bars")
    print("model card updated: https://huggingface.co/" + REPO_ID)

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "2003Rohansinha/Qwen2.5-Coder-1.5B-Python-OCI-LoRA"   # change if your REPO_NAME differs
SYSTEM = "You are an expert Python programmer. Answer with correct, complete, runnable Python code."

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
model.eval()

def generate(question, max_new_tokens):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=int(max_new_tokens), do_sample=False)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

demo = gr.Interface(
    fn=generate,
    inputs=[gr.Textbox(lines=4, label="Describe the Python code you want"),
            gr.Slider(64, 512, value=256, step=32, label="Max new tokens")],
    outputs=gr.Markdown(label="Answer"),
    title="Qwen2.5-Coder-1.5B Python (OpenCodeInstruct LoRA)",
    description="Small model: always review and test generated code. Runs on free CPU, so answers take a minute or so.",
    examples=[["Write a function that merges two sorted lists.", 256],
              ["Write a function that returns the k most frequent words in a text file.", 256]],
    allow_flagging="never",
)
demo.launch()

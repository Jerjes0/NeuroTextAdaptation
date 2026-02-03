# keyword_filter.py
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# -------- CONFIG --------
PARQUET_PATH = ""
OUT_CSV = "keywords_out.csv"
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
N_ROWS = 20
TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
# ------------------------

LABEL_RE = re.compile(r"^(terms|keywords|medical terms|relational words)\s*:\s*", re.I)

def extract_comma_list(text: str) -> str:
    text = "" if text is None else str(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    comma_lines = [ln for ln in lines if "," in ln]
    candidate = comma_lines[-1] if comma_lines else (lines[-1] if lines else "")
    return LABEL_RE.sub("", candidate).strip()

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    trust_remote_code=True,
).to(device)

def run_llm(title: str, abstract: str, temperature: float) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Extract medical terms and relational words. "
                "Return ONE LINE, comma-separated. No labels."
            ),
        },
        {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    enc = tokenizer(prompt, return_tensors="pt").to(device)

    kwargs = dict(
        max_new_tokens=128,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        kwargs.update(do_sample=False)

    out = model.generate(**enc, **kwargs)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)

    return extract_comma_list(decoded)

print("Loading cleaned parquet:", PARQUET_PATH)
df = pd.read_parquet(PARQUET_PATH).head(N_ROWS)

rows = []
for i, r in df.iterrows():
    rec = {
        "index": i,
        "title": r["title_clean"],
        "abstract": r["abstract_clean"],
    }

    for t in TEMPERATURES:
        rec[f"mediphi_temp_{t}"] = run_llm(
            r["title_clean"], r["abstract_clean"], t
        )

    rows.append(rec)

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CSV, index=False)
print("✅ Saved:", OUT_CSV)
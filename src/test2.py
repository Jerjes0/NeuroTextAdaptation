import re
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------- CONFIG ----------------
PARQUET_PATH = "data/abstracts.parquet"
OUTPUT_CSV = "mediphi100T.csv"
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
N_ROWS = 100
TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
# ---------------------------------------

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Fix attention_mask warning when pad == eos
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=None,
    torch_dtype="auto",
    trust_remote_code=True,
).to(device)

def clean_html(s: str) -> str:
    s = "" if s is None else str(s)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def extract_comma_list(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    comma_lines = [ln for ln in lines if "," in ln]
    candidate = comma_lines[-1] if comma_lines else (lines[-1] if lines else "")
    candidate = re.sub(r"^(terms|keywords|medical terms|relational words)\s*:\s*", "", candidate, flags=re.I)
    return candidate.strip()

def extract_terms_mediphi(title: str, abstract: str, temperature: float) -> str:
    title = clean_html(title)
    abstract = clean_html(abstract)

    combined_text = f"Title: {title}\nAbstract: {abstract}"

    messages = [
        {
            "role": "system",
            "content": (
                "Extract medical terms and relational words from the user's text. "
                "Return ONLY ONE LINE: a comma-separated list of extracted words/phrases. "
                "No explanations, no labels. Do NOT include the words 'Title' or 'Abstract'."
            ),
        },
        {"role": "user", "content": combined_text},
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    enc = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    gen_kwargs = dict(
        max_new_tokens=128,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    if temperature > 0:
        gen_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
    else:
        gen_kwargs.update(do_sample=False)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Remove echoed prompt if present
    if decoded.startswith(input_text):
        decoded = decoded[len(input_text):].strip()

    # Keep last useful line (often the comma list)
    return extract_comma_list(decoded)

# Load parquet + first N rows
df = pd.read_parquet(PARQUET_PATH).head(N_ROWS)

results = []
for i, row in df.iterrows():
    title = "" if pd.isna(row.get("title")) else row.get("title")
    abstract = "" if pd.isna(row.get("abstract")) else row.get("abstract")

    record = {
        "index": i,
        "title": clean_html(title),
        "abstract": clean_html(abstract),
    }

    for temp in TEMPERATURES:
        record[f"mediphi_temp_{temp}"] = extract_terms_mediphi(title, abstract, temp)

    results.append(record)

out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(out_df)} rows to {OUTPUT_CSV}")
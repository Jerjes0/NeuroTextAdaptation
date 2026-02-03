import re
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------- CONFIG ----------------
PARQUET_PATH = "/Users/alexis/Desktop/NeuroTextAdapt/NeuroTextAdaptation/data/abstracts.parquet"
OUTPUT_CSV = "filter2.csv"
CHECKPOINT = "microsoft/MediPhi"
N_ROWS = 15
TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
# ---------------------------------------

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

# ---- FIX pad token / attention mask warning ----
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(CHECKPOINT).to(device)

def clean_html(s: str) -> str:
    s = "" if s is None else str(s)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def extract_comma_list(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    comma_lines = [ln for ln in lines if "," in ln]
    candidate = comma_lines[-1] if comma_lines else (lines[-1] if lines else "")
    candidate = re.sub(
        r"^(terms|keywords|medical terms|relational words)\s*:\s*",
        "",
        candidate,
        flags=re.I,
    )
    return candidate.strip()

def run_smolllm2(title: str, abstract: str, temperature: float) -> str:
    title = clean_html(title)
    abstract = clean_html(abstract)

    prompt = (
        "You are a professional in the neuroscience and medicine field and you are tasked to read titles and abstract and extract the most important keywords from them."
        "You are extracting terms that are also relational to eachother, specifically words that are essential for each title and abstract..\n"
        "More importantly I do not want a summary of the title/abstract i want a list of words that are important"
        "Output format MUST be:\n"
        "TERMS: item1, item2, item3\n"
        "Rules: ONLY one line. ONLY commas. NO extra words.\n\n"
        "Example: " \
        "example_title:Network neuroscience" \
        "example_abstract:Despite substantial recent progress, our understanding of the principles and mechanisms underlying complex brain function and cognition remains incomplete. Network neuroscience proposes to tackle these enduring challenges. Approaching brain structure and function from an explicitly integrative perspective, network neuroscience pursues new ways to map, record, analyze and model the elements and interactions of neurobiological systems. Two parallel trends drive the approach: the availability of new empirical tools to create comprehensive maps and record dynamic patterns among molecules, neurons, brain areas and social systems; and the theoretical framework and computational tools of modern network science. The convergence of empirical and computational advances opens new frontiers of scientific inquiry, including network dynamics, manipulation and control of brain networks, and integration of network processes across spatiotemporal domains. We review emerging trends in network neuroscience and attempt to chart a path toward a better understanding of the brain as a multiscale networked system.:" \
        "example_output = understanding, principles, mechanisms, cognition, neuroscience, network, neurobiological, dynamics, spatiotemporal"
        f"TITLE: {title}\n"
        f"ABSTRACT: {abstract}\n"
        "TERMS:"
    )

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)

    enc = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=128, # potential issue
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else None,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt echo if present
    if decoded.startswith(input_text):
        decoded = decoded[len(input_text):]

    if "TERMS:" in decoded:
        decoded = decoded.split("TERMS:")[-1]

    return extract_comma_list(decoded.strip())

# -------- Load parquet --------
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
        record[f"smolllm2_temp_{temp}"] = run_smolllm2(title, abstract, temp)

    results.append(record)

out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved {len(out_df)} rows to {OUTPUT_CSV}")
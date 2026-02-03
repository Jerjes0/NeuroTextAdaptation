import re
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

# ---------------- CONFIG ----------------
PARQUET_PATH = "/Users/alexis/Desktop/NeuroTextAdapt/NeuroTextAdaptation/data/abstracts.parquet"
OUTPUT_CSV = "filter9.csv"
CHECKPOINT = "HuggingFaceTB/SmolLM2-135M-Instruct"
N_ROWS = 100
TEMPERATURES = [0.5]
MAX_NEW_TOKENS = 128
# ---------------------------------------

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

# Fix pad token / attention mask warning
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(CHECKPOINT).to(device)
model.eval()

# ---------- SAFE CLEANING ----------

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def safe_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, (list, tuple, set)):
        return " ".join(safe_text(v) for v in x)
    if isinstance(x, dict):
        return " ".join(f"{k}:{safe_text(v)}" for k, v in x.items())
    return str(x)

def clean_text(s: str) -> str:
    s = safe_text(s)
    if not s:
        return ""
    s = TAG_RE.sub(" ", s)
    s = CTRL_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s

# ---------- OUTPUT PARSING ----------

LABEL_RE = re.compile(r"^(terms|keywords|medical terms|relational words)\s*:\s*", re.I)

def extract_comma_list(text: str) -> str:
    text = safe_text(text).strip()
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    comma_lines = [ln for ln in lines if "," in ln]
    candidate = comma_lines[-1] if comma_lines else lines[-1]
    candidate = candidate.strip(" -*\t\r")
    return LABEL_RE.sub("", candidate).strip()

# ---------- LLM CALL ----------

def run_keyword(title_clean: str, abstract_clean: str, temperature: float) -> str:
    prompt = (
        "You are a professional in the neuroscience and medicine field and you are tasked to read titles and abstract and extract the most important keywords from them."
        "You are extracting terms that are also relational to each other, specifically words that are essential for each title and abstract.\n"
        "More importantly I do not want a summary of the title/abstract i want a list of words that are important"
        "Output format MUST be:\n"
        "TERMS: item1, item2, item3\n"
        "Rules: ONLY one line. ONLY commas. NO extra words.\n\n"
        "Example: " \
        "example_title:Network neuroscience" \
        "example_abstract:Despite substantial recent progress, our understanding of the principles and mechanisms underlying complex brain function and cognition remains incomplete. Network neuroscience proposes to tackle these enduring challenges. Approaching brain structure and function from an explicitly integrative perspective, network neuroscience pursues new ways to map, record, analyze and model the elements and interactions of neurobiological systems. Two parallel trends drive the approach: the availability of new empirical tools to create comprehensive maps and record dynamic patterns among molecules, neurons, brain areas and social systems; and the theoretical framework and computational tools of modern network science. The convergence of empirical and computational advances opens new frontiers of scientific inquiry, including network dynamics, manipulation and control of brain networks, and integration of network processes across spatiotemporal domains. We review emerging trends in network neuroscience and attempt to chart a path toward a better understanding of the brain as a multiscale networked system.:" \
        "example_output = "
        ""
        "TERMS: understanding, principles, mechanisms, cognition"
        f"TITLE: {title_clean}\n"
        f"ABSTRACT: {abstract_clean}\n"
        "TERMS:"
    )

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    enc = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        outputs = model.generate(**enc, **gen_kwargs)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if decoded.startswith(input_text):
        decoded = decoded[len(input_text):].strip()

    if "TERMS:" in decoded:
        decoded = decoded.split("TERMS:")[-1].strip()

    return extract_comma_list(decoded)

# ---------- LOAD DATA ----------

df = pd.read_parquet(PARQUET_PATH).head(N_ROWS)

results = []
emptied = 0

print(f"Processing {len(df)} rows...")

for i, row in tqdm(df.iterrows(), total=len(df), desc="Running filter"):
    title_raw = safe_text(row.get("title"))
    abstract_raw = safe_text(row.get("abstract"))

    title_clean = clean_text(title_raw)
    abstract_clean = clean_text(abstract_raw)

    if (title_raw.strip() and not title_clean) or (abstract_raw.strip() and not abstract_clean):
        emptied += 1

    record = {
        "index": i,
        "title_raw": title_raw,
        "abstract_raw": abstract_raw,
        "title_clean": title_clean,
        "abstract_clean": abstract_clean,
    }

    for temp in TEMPERATURES:
        record[f"mediphi_temp_{temp}"] = run_keyword(
            title_clean, abstract_clean, temp
        )

    results.append(record)


# ---------- SAVE ----------

out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Saved {len(out_df)} rows → {OUTPUT_CSV}")
print(f"⚠️ Cleaning emptied non-empty rows: {emptied}/{len(out_df)}")
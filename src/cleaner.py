# clean_parquet.py
import re
import pandas as pd

# -------- CONFIG --------
IN_PARQUET = "/Users/alexis/Desktop/NeuroTextAdapt/NeuroTextAdaptation/data/abstracts.parquet"
OUT_PARQUET = "data/abstracts.cleaned.parquet"
# ------------------------

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

def clean_html(s: str) -> str:
    s = "" if s is None else str(s)
    s = TAG_RE.sub(" ", s)
    return WS_RE.sub(" ", s).strip()

def main():
    print("Reading:", IN_PARQUET)
    df = pd.read_parquet(IN_PARQUET)

    # Expect columns: title, abstract
    df["title_clean"] = df["title"].apply(clean_html)
    df["abstract_clean"] = df["abstract"].apply(clean_html)

    # Keep originals if you want; otherwise drop them
    out = df[["title_clean", "abstract_clean"]]

    print("Writing:", OUT_PARQUET)
    out.to_parquet(OUT_PARQUET, index=False)

    print("✅ Cleaning complete")
    print("Rows:", len(out))

if __name__ == "__main__":
    main()
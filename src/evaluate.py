import argparse
import csv
import json
import logging
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
# Folder containing predictions.csv from predict.py
PREDICTIONS_DIR = Path("outputs/predict-finetuned-noctx-20260217-153151")
PREDICTIONS_CSV_PATH = PREDICTIONS_DIR / "predictions.csv"

# Input CSV columns
REFERENCE_COL = "answer"
PREDICTION_COL = "predicted_answer"

# BERTScore settings
BERTSCORE_MODEL_TYPE = "distilbert-base-uncased"
BERTSCORE_BATCH_SIZE = 16
BERTSCORE_DEVICE = "cpu"

RUN_NAME = f"evaluate-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
LOG_PATH = PREDICTIONS_DIR / "evaluate.log"
RUN_CONFIG_PATH = PREDICTIONS_DIR / "evaluate_run_config.json"
PER_EXAMPLE_METRICS_PATH = PREDICTIONS_DIR / "evaluation_metrics.csv"
SUMMARY_METRICS_PATH = PREDICTIONS_DIR / "evaluation_summary.csv"


def setup_logging(log_path: Path, logger_name: str = "pubmed-eval") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger(logger_name)


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


def _ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(reference: str, prediction: str, max_order: int = 4) -> float:
    ref_toks = _tokenize(reference)
    pred_toks = _tokenize(prediction)
    if not ref_toks or not pred_toks:
        return 0.0

    effective_order = min(max_order, len(ref_toks), len(pred_toks))
    if effective_order == 0:
        return 0.0

    log_precision_sum = 0.0
    for n in range(1, effective_order + 1):
        ref_counts = _ngram_counts(ref_toks, n)
        pred_counts = _ngram_counts(pred_toks, n)
        overlap = sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
        total = sum(pred_counts.values())
        smoothed_precision = (overlap + 1.0) / (total + 1.0)
        log_precision_sum += math.log(smoothed_precision)

    geo_mean = math.exp(log_precision_sum / effective_order)
    ref_len, pred_len = len(ref_toks), len(pred_toks)
    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1.0 - (ref_len / pred_len))
    return float(brevity_penalty * geo_mean)


def rouge_n_f1(reference: str, prediction: str, n: int) -> float:
    ref_toks = _tokenize(reference)
    pred_toks = _tokenize(prediction)
    ref_counts = _ngram_counts(ref_toks, n)
    pred_counts = _ngram_counts(pred_toks, n)
    if not ref_counts or not pred_counts:
        return 0.0

    overlap = sum(min(count, pred_counts[gram]) for gram, count in ref_counts.items())
    precision = overlap / sum(pred_counts.values())
    recall = overlap / sum(ref_counts.values())
    if precision + recall == 0:
        return 0.0
    return float((2 * precision * recall) / (precision + recall))


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, start=1):
            tmp = dp[j]
            if x == y:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref_toks = _tokenize(reference)
    pred_toks = _tokenize(prediction)
    if not ref_toks or not pred_toks:
        return 0.0
    lcs = _lcs_length(ref_toks, pred_toks)
    precision = lcs / len(pred_toks)
    recall = lcs / len(ref_toks)
    if precision + recall == 0:
        return 0.0
    return float((2 * precision * recall) / (precision + recall))


def token_f1(reference: str, prediction: str) -> float:
    ref_toks = _tokenize(reference)
    pred_toks = _tokenize(prediction)
    if not ref_toks or not pred_toks:
        return 0.0
    common = Counter(ref_toks) & Counter(pred_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(ref_toks)
    return float((2 * precision * recall) / (precision + recall))


def jaccard_similarity(reference: str, prediction: str) -> float:
    ref_set = set(_tokenize(reference))
    pred_set = set(_tokenize(prediction))
    if not ref_set and not pred_set:
        return 1.0
    union = ref_set | pred_set
    if not union:
        return 0.0
    return float(len(ref_set & pred_set) / len(union))


def compute_tfidf_cosine_per_row(
    references: list[str], predictions: list[str], log
) -> tuple[list[float], bool]:
    if not references:
        return [], False
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        log.warning("scikit-learn or numpy not installed; tfidf_cosine column will be NaN.")
        return [float("nan")] * len(references), False

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        matrix = vectorizer.fit_transform(references + predictions)
        n = len(references)
        ref_mat = matrix[:n]
        pred_mat = matrix[n:]

        numerator = ref_mat.multiply(pred_mat).sum(axis=1).A1
        ref_norm = np.sqrt(ref_mat.multiply(ref_mat).sum(axis=1).A1)
        pred_norm = np.sqrt(pred_mat.multiply(pred_mat).sum(axis=1).A1)
        denom = ref_norm * pred_norm

        cosines = np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom > 0)
        return [float(x) for x in cosines], True
    except Exception as err:
        log.warning("TF-IDF cosine computation failed (%s); column will be NaN.", err)
        return [float("nan")] * len(references), False


def maybe_compute_bertscore(
    references: list[str], predictions: list[str], log
) -> tuple[list[float], list[float], list[float], bool]:
    try:
        from bert_score import score as bertscore_score
    except ImportError:
        log.warning("bert-score package not installed; BERTScore columns will be NaN.")
        nan_values = [float("nan")] * len(references)
        return nan_values, nan_values, nan_values, False

    try:
        p, r, f1 = bertscore_score(
            cands=predictions,
            refs=references,
            model_type=BERTSCORE_MODEL_TYPE,
            batch_size=BERTSCORE_BATCH_SIZE,
            device=BERTSCORE_DEVICE,
            verbose=True,
            lang="en",
        )
        return p.tolist(), r.tolist(), f1.tolist(), True
    except Exception as err:
        log.warning("BERTScore computation failed (%s); columns will be NaN.", err)
        nan_values = [float("nan")] * len(references)
        return nan_values, nan_values, nan_values, False


def safe_mean(values: list[float]) -> float:
    clean = [v for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return float(statistics.fmean(clean)) if clean else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions.csv with text-generation metrics.")
    parser.add_argument(
        "--predictions_dir",
        type=Path,
        default=PREDICTIONS_DIR,
        help="Folder containing predictions.csv (default from config).",
    )
    args_ns = parser.parse_args()

    predictions_dir = args_ns.predictions_dir
    predictions_csv_path = predictions_dir / "predictions.csv"
    log_path = predictions_dir / "evaluate.log"
    run_config_path = predictions_dir / "evaluate_run_config.json"
    per_example_path = predictions_dir / "evaluation_metrics.csv"
    summary_path = predictions_dir / "evaluation_summary.csv"

    if not predictions_csv_path.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {predictions_csv_path}")

    log = setup_logging(log_path, logger_name="pubmed-eval")
    run_config = {
        "run_name": RUN_NAME,
        "predictions_dir": str(predictions_dir),
        "predictions_csv_path": str(predictions_csv_path),
        "reference_col": REFERENCE_COL,
        "prediction_col": PREDICTION_COL,
        "bertscore_model_type": BERTSCORE_MODEL_TYPE,
        "bertscore_batch_size": BERTSCORE_BATCH_SIZE,
        "bertscore_device": BERTSCORE_DEVICE,
        "per_example_metrics_path": str(per_example_path),
        "summary_metrics_path": str(summary_path),
    }
    run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    log.info("Run config saved to %s", run_config_path)

    rows: list[dict[str, Any]] = []
    with predictions_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing_cols = {REFERENCE_COL, PREDICTION_COL} - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(f"Missing required columns in predictions CSV: {sorted(missing_cols)}")
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("predictions.csv has no rows to evaluate.")

    references = [str(r.get(REFERENCE_COL, "")).strip() for r in rows]
    predictions = [str(r.get(PREDICTION_COL, "")).strip() for r in rows]

    tfidf_cos, tfidf_ok = compute_tfidf_cosine_per_row(references, predictions, log)
    b_p, b_r, b_f1, bertscore_ok = maybe_compute_bertscore(references, predictions, log)
    log.info("TF-IDF cosine enabled: %s", tfidf_ok)
    log.info("BERTScore enabled: %s", bertscore_ok)

    metric_rows: list[dict[str, Any]] = []
    bleu_vals: list[float] = []
    rouge1_vals: list[float] = []
    rouge2_vals: list[float] = []
    rougel_vals: list[float] = []
    em_vals: list[float] = []
    token_f1_vals: list[float] = []
    jaccard_vals: list[float] = []
    length_ratio_vals: list[float] = []

    for i, row in enumerate(rows):
        ref = references[i]
        pred = predictions[i]

        em = float(normalize_text(ref) == normalize_text(pred))
        bleu = sentence_bleu(ref, pred)
        rouge1 = rouge_n_f1(ref, pred, n=1)
        rouge2 = rouge_n_f1(ref, pred, n=2)
        rougel = rouge_l_f1(ref, pred)
        tok_f1 = token_f1(ref, pred)
        jac = jaccard_similarity(ref, pred)
        ref_len = len(_tokenize(ref))
        pred_len = len(_tokenize(pred))
        length_ratio = float(pred_len / ref_len) if ref_len > 0 else float("nan")

        bleu_vals.append(bleu)
        rouge1_vals.append(rouge1)
        rouge2_vals.append(rouge2)
        rougel_vals.append(rougel)
        em_vals.append(em)
        token_f1_vals.append(tok_f1)
        jaccard_vals.append(jac)
        length_ratio_vals.append(length_ratio)

        metric_rows.append(
            {
                "index": row.get("index", i),
                "pmid": row.get("pmid", ""),
                "question": row.get("question", ""),
                "exact_match": em,
                "bleu": bleu,
                "rouge1_f1": rouge1,
                "rouge2_f1": rouge2,
                "rougeL_f1": rougel,
                "token_f1": tok_f1,
                "jaccard": jac,
                "tfidf_cosine": tfidf_cos[i],
                "bertscore_precision": b_p[i],
                "bertscore_recall": b_r[i],
                "bertscore_f1": b_f1[i],
                "reference_len_tokens": ref_len,
                "prediction_len_tokens": pred_len,
                "length_ratio_pred_over_ref": length_ratio,
            }
        )

    with per_example_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    summary_row = {
        "num_examples": len(metric_rows),
        "exact_match_mean": safe_mean(em_vals),
        "bleu_mean": safe_mean(bleu_vals),
        "rouge1_f1_mean": safe_mean(rouge1_vals),
        "rouge2_f1_mean": safe_mean(rouge2_vals),
        "rougeL_f1_mean": safe_mean(rougel_vals),
        "token_f1_mean": safe_mean(token_f1_vals),
        "jaccard_mean": safe_mean(jaccard_vals),
        "tfidf_cosine_mean": safe_mean(tfidf_cos),
        "bertscore_precision_mean": safe_mean(b_p),
        "bertscore_recall_mean": safe_mean(b_r),
        "bertscore_f1_mean": safe_mean(b_f1),
        "length_ratio_mean": safe_mean(length_ratio_vals),
    }

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    log.info("Saved per-example metrics to %s", per_example_path)
    log.info("Saved summary metrics to %s", summary_path)
    log.info("Summary: %s", summary_row)


if __name__ == "__main__":
    main()

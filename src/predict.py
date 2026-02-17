import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import wandb
from datasets import load_dataset, load_from_disk
from utils import add_nearest_context
from utils import ensure_dirs
from utils import explode_qa_pairs_with_prompt
from utils import get_required_env
from utils import get_selected_split
from utils import load_prompt_template
from utils import load_prediction_model
from utils import normalize_text
from utils import setup_logging
from utils import split_pmids

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DATASET_ID = '/Users/jerjesaguirrechavez/Desktop/UCSD/Voytek Lab/NeuroTextAdaptation/data/filtered_dataset_with_vectors' #"Jerjes/pubmed_summary_qa_w_encodings"
# If True, DATASET_ID is interpreted as a Hugging Face dataset repo id.
# If False, DATASET_ID is interpreted as a local dataset path for load_from_disk.
LOAD_FROM_HF = False

# "zero_shot" uses ZERO_SHOT_MODEL_ID.
# "finetuned" uses FINETUNED_MODEL_PATH (adapter or full model depending on FINETUNED_IS_ADAPTER).
MODEL_MODE = "finetuned"  # "zero_shot" | "finetuned"

WITH_CONTEXT = False

ZERO_SHOT_MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
FINETUNED_MODEL_PATH = "Jerjes/smolllm2-pubmed-qa-noctx-20260217-152143"
FINETUNED_IS_ADAPTER = True

RANDOM_SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)
MAX_PMIDS_DEFAULT = -1 # all of it
MAX_PMIDS = None

PREDICT_SPLIT = "test"  # "train" | "val" | "test"
NUM_CONTEXT_PAPERS = 3
MAX_PREDICT_EXAMPLES = -1  # -1 means all examples in selected split

MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 256
DO_SAMPLE = False
TEMPERATURE = 0.2
TOP_P = 0.9
USE_4BIT = True
LOG_EVERY = 100

WANDB_REQUIRED = True
WANDB_PROJECT = "neurotextadaptation-predict"

PROMPT_WITH_CONTEXT_PATH = Path("src/prompts/qa_context_prompt.txt")
PROMPT_NO_CONTEXT_PATH = Path("src/prompts/qa_no_context_prompt.txt")

CONTEXT_TAG = "withctx" if WITH_CONTEXT else "noctx"
RUN_NAME = f"predict-{MODEL_MODE}-{CONTEXT_TAG}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
OUTPUT_ROOT = Path("outputs")
OUTPUT_DIR = OUTPUT_ROOT / RUN_NAME
LOG_PATH = OUTPUT_DIR / "predict.log"
RUN_CONFIG_PATH = OUTPUT_DIR / "run_config.json"
PREDICTIONS_CSV_PATH = OUTPUT_DIR / "predictions.csv"
METRICS_PATH = OUTPUT_DIR / "predict_metrics.json"


def log_pipeline_event(log, stage: str, stage_idx: int, **metrics: float) -> None:
    log.info("Pipeline stage %d: %s", stage_idx, stage)
    if metrics:
        log.info("Stage metrics (%s): %s", stage, metrics)
    if wandb.run is not None:
        payload: dict[str, float] = {"pipeline/stage_index": float(stage_idx)}
        payload.update({f"pipeline/{k}": v for k, v in metrics.items()})
        wandb.log(payload)
        wandb.run.summary["pipeline/last_stage"] = stage


def main() -> None:
    global MAX_PMIDS

    parser = argparse.ArgumentParser(description="Prediction runner for pubmed_summary_qa")
    parser.add_argument("--max_pmids", type=int, default=MAX_PMIDS, help="Override number of PMIDs (None uses dev default).")
    parser.add_argument("--max_predict_examples", type=int, default=MAX_PREDICT_EXAMPLES, help="Cap prediction examples (-1 means all).")
    args_ns = parser.parse_args()

    if args_ns.max_pmids is not None:
        MAX_PMIDS = args_ns.max_pmids
    max_predict_examples = args_ns.max_predict_examples

    ensure_dirs(OUTPUT_DIR)
    log = setup_logging(LOG_PATH, logger_name="pubmed-predict")
    prompt_path = PROMPT_WITH_CONTEXT_PATH if WITH_CONTEXT else PROMPT_NO_CONTEXT_PATH
    prompt_template = load_prompt_template(prompt_path, require_context=WITH_CONTEXT)

    hf_token = get_required_env("HF_TOKEN")
    if WANDB_REQUIRED:
        get_required_env("WANDB_API_KEY")

    run_config = {
        "dataset_id": DATASET_ID,
        "load_from_hf": LOAD_FROM_HF,
        "model_mode": MODEL_MODE,
        "zero_shot_model_id": ZERO_SHOT_MODEL_ID,
        "finetuned_model_path": FINETUNED_MODEL_PATH,
        "finetuned_is_adapter": FINETUNED_IS_ADAPTER,
        "predict_split": PREDICT_SPLIT,
        "with_context": WITH_CONTEXT,
        "prompt_path": str(prompt_path),
        "random_seed": RANDOM_SEED,
        "split_ratios": {"train": SPLIT_RATIOS[0], "val": SPLIT_RATIOS[1], "test": SPLIT_RATIOS[2]},
        "max_pmids_default": MAX_PMIDS_DEFAULT,
        "max_pmids_effective": MAX_PMIDS if MAX_PMIDS is not None else MAX_PMIDS_DEFAULT,
        "num_context_papers": NUM_CONTEXT_PAPERS,
        "max_predict_examples": max_predict_examples,
        "generation": {
            "max_seq_length": MAX_SEQ_LENGTH,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": DO_SAMPLE,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "use_4bit": USE_4BIT,
        },
        "output_dir": str(OUTPUT_DIR),
        "predictions_csv_path": str(PREDICTIONS_CSV_PATH),
    }
    RUN_CONFIG_PATH.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    log.info("Run config saved to %s", RUN_CONFIG_PATH)
    log.info("Run config:\n%s", json.dumps(run_config, indent=2))

    wandb.init(project=WANDB_PROJECT, name=RUN_NAME, config=run_config)
    stage_idx = 1
    log_pipeline_event(log, "run_config_saved", stage_idx)

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    log.info("Loading dataset: %s (load_from_hf=%s)", DATASET_ID, LOAD_FROM_HF)
    if LOAD_FROM_HF:
        raw_dataset = load_dataset(DATASET_ID, split="train")
    else:
        raw_dataset = load_from_disk(DATASET_ID)
    unique_pmids = [int(x) for x in raw_dataset.unique("pmid")]
    stage_idx += 1
    log_pipeline_event(
        log,
        "dataset_loaded",
        stage_idx,
        dataset_rows=float(len(raw_dataset)),
        unique_pmids=float(len(unique_pmids)),
        load_seconds=round(time.perf_counter() - t0, 3),
    )

    # ------------------------------------------------------------
    # Deterministic PMID-level split
    # ------------------------------------------------------------
    train_pmids, val_pmids, test_pmids = split_pmids(
        unique_pmids=unique_pmids,
        seed=RANDOM_SEED,
        split_ratios=SPLIT_RATIOS,
        max_pmids=MAX_PMIDS,
        max_pmids_default=MAX_PMIDS_DEFAULT,
    )
    if train_pmids & val_pmids or train_pmids & test_pmids or val_pmids & test_pmids:
        raise RuntimeError("PMID leakage detected across splits.")
    stage_idx += 1
    log_pipeline_event(
        log,
        "pmid_split_done",
        stage_idx,
        train_pmids=float(len(train_pmids)),
        val_pmids=float(len(val_pmids)),
        test_pmids=float(len(test_pmids)),
    )

    t0 = time.perf_counter()
    selected_raw = get_selected_split(
        raw_dataset=raw_dataset,
        predict_split=PREDICT_SPLIT,
        train_pmids=train_pmids,
        val_pmids=val_pmids,
        test_pmids=test_pmids,
    )
    stage_idx += 1
    log_pipeline_event(
        log,
        "split_selected",
        stage_idx,
        split_rows=float(len(selected_raw)),
        split_seconds=round(time.perf_counter() - t0, 3),
    )

    # ------------------------------------------------------------
    # Build retrieval context and prediction examples
    # ------------------------------------------------------------
    if WITH_CONTEXT:
        t0 = time.perf_counter()
        selected_raw = add_nearest_context(selected_raw, top_k=NUM_CONTEXT_PAPERS, log=log)
        stage_idx += 1
        log_pipeline_event(
            log,
            "retrieval_context_ready",
            stage_idx,
            context_k=float(NUM_CONTEXT_PAPERS),
            rows_with_context=float(len(selected_raw)),
            retrieval_seconds=round(time.perf_counter() - t0, 3),
        )
    else:
        stage_idx += 1
        log_pipeline_event(
            log,
            "retrieval_context_skipped",
            stage_idx,
            with_context=0.0,
            rows=float(len(selected_raw)),
        )

    t0 = time.perf_counter()
    predict_dataset = explode_qa_pairs_with_prompt(selected_raw, prompt_template=prompt_template)
    if max_predict_examples > 0:
        predict_dataset = predict_dataset.select(range(min(max_predict_examples, len(predict_dataset))))
    stage_idx += 1
    log_pipeline_event(
        log,
        "prediction_examples_built",
        stage_idx,
        predict_examples=float(len(predict_dataset)),
        build_seconds=round(time.perf_counter() - t0, 3),
    )

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    model, tokenizer, use_bf16, infer_device, use_4bit_effective = load_prediction_model(
        hf_token=hf_token,
        model_mode=MODEL_MODE,
        zero_shot_model_id=ZERO_SHOT_MODEL_ID,
        finetuned_model_path=FINETUNED_MODEL_PATH,
        finetuned_is_adapter=FINETUNED_IS_ADAPTER,
        use_4bit=USE_4BIT,
    )
    model.eval()
    stage_idx += 1
    log_pipeline_event(
        log,
        "model_loaded",
        stage_idx,
        bf16_enabled=float(int(use_bf16)),
        use_4bit=float(int(use_4bit_effective)),
        finetuned_mode=float(int(MODEL_MODE == "finetuned")),
    )
    log.info("Inference device: %s | effective_4bit=%s", infer_device, use_4bit_effective)

    # ------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------
    exact_matches = 0
    total = len(predict_dataset)
    t0 = time.perf_counter()

    with PREDICTIONS_CSV_PATH.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "index",
                "pmid",
                "doi",
                "title",
                "abstract",
                "question",
                "answer",
                "context",
                "predicted_answer",
                "exact_match",
            ],
        )
        writer.writeheader()
        csv_file.flush()

        for i, example in enumerate(predict_dataset):
            user_prompt = example["messages"][0]["content"]
            question = str(example.get("question", "")).strip()
            context = str(example.get("context_block", "")).strip()
            target_answer = str(example.get("answer", "")).strip()
            infer_messages = [{"role": "user", "content": user_prompt}]

            input_text = tokenizer.apply_chat_template(
                infer_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            )
            enc = {k: v.to(infer_device) for k, v in enc.items()}

            gen_kwargs = {
                "max_new_tokens": MAX_NEW_TOKENS,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": DO_SAMPLE,
            }
            if DO_SAMPLE:
                gen_kwargs["temperature"] = TEMPERATURE
                gen_kwargs["top_p"] = TOP_P

            with torch.no_grad():
                output = model.generate(**enc, **gen_kwargs)

            generated_tokens = output[0][enc["input_ids"].shape[1] :]
            prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            is_exact_match = int(normalize_text(prediction) == normalize_text(target_answer))
            exact_matches += is_exact_match

            writer.writerow(
                {
                    "index": i,
                    "pmid": example.get("pmid"),
                    "doi": example.get("doi"),
                    "title": example.get("title"),
                    "abstract": example.get("summary"),
                    "question": question,
                    "answer": target_answer,
                    "context": context,
                    "predicted_answer": prediction,
                    "exact_match": is_exact_match,
                }
            )
            csv_file.flush()

            if (i + 1) % LOG_EVERY == 0:
                partial_em = exact_matches / (i + 1)
                log.info("Predicted %d/%d examples | partial_exact_match=%.4f", i + 1, len(predict_dataset), partial_em)
                if wandb.run is not None:
                    wandb.log(
                        {
                            "predict/processed_examples": float(i + 1),
                            "predict/partial_exact_match": partial_em,
                        }
                    )

    exact_match_rate = (exact_matches / total) if total > 0 else 0.0
    elapsed = time.perf_counter() - t0

    metrics = {
        "predict_examples": total,
        "exact_match_count": exact_matches,
        "exact_match_rate": exact_match_rate,
        "predict_seconds": round(elapsed, 3),
        "predict_examples_per_second": round((total / elapsed) if elapsed > 0 else 0.0, 3),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Prediction metrics: %s", metrics)
    log.info("Saved predictions to %s", PREDICTIONS_CSV_PATH)
    log.info("Saved metrics to %s", METRICS_PATH)

    stage_idx += 1
    log_pipeline_event(
        log,
        "prediction_finished",
        stage_idx,
        predict_examples=float(total),
        exact_match_rate=exact_match_rate,
        predict_seconds=round(elapsed, 3),
    )
    if wandb.run is not None:
        wandb.log(
            {
                "predict/final_examples": float(total),
                "predict/final_exact_match_rate": exact_match_rate,
                "predict/seconds": round(elapsed, 3),
            }
        )
        wandb.run.summary["predict/examples"] = total
        wandb.run.summary["predict/exact_match_rate"] = exact_match_rate

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------
    del model
    torch.cuda.empty_cache()
    stage_idx += 1
    log_pipeline_event(log, "cleanup_finished", stage_idx)
    wandb.finish()


if __name__ == "__main__":
    main()

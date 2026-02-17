import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import wandb
from datasets import load_dataset, load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils import add_nearest_context
from utils import ensure_dirs
from utils import explode_qa_pairs_with_prompt
from utils import get_required_env
from utils import load_prompt_template
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
MODEL_MODE = "zero_shot"  # "zero_shot" | "finetuned"
ZERO_SHOT_MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
FINETUNED_MODEL_PATH = "your-username/your-finetuned-model-or-adapter"
FINETUNED_IS_ADAPTER = True

RANDOM_SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)
MAX_PMIDS_DEFAULT = 10000
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
PROMPT_PATH = Path("src/prompts/qa_context_prompt.txt")

RUN_NAME = f"predict-{MODEL_MODE}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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


def get_selected_split(raw_dataset, train_pmids: set[int], val_pmids: set[int], test_pmids: set[int]):
    if PREDICT_SPLIT == "train":
        return raw_dataset.filter(lambda x: int(x["pmid"]) in train_pmids, desc="Filtering train split")
    if PREDICT_SPLIT == "val":
        return raw_dataset.filter(lambda x: int(x["pmid"]) in val_pmids, desc="Filtering val split")
    if PREDICT_SPLIT == "test":
        return raw_dataset.filter(lambda x: int(x["pmid"]) in test_pmids, desc="Filtering test split")
    raise ValueError("PREDICT_SPLIT must be one of: train, val, test")


def load_prediction_model(hf_token: str):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer_model_id = ZERO_SHOT_MODEL_ID if MODEL_MODE == "finetuned" else ZERO_SHOT_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_bf16 = bool(device.type == "cuda" and torch.cuda.is_bf16_supported())
    if device.type == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if use_bf16 else torch.float16

    use_4bit_effective = bool(USE_4BIT and device.type == "cuda")
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if use_4bit_effective:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    if MODEL_MODE == "zero_shot":
        model = AutoModelForCausalLM.from_pretrained(ZERO_SHOT_MODEL_ID, token=hf_token, **model_kwargs)
        if not use_4bit_effective:
            model.to(device)
        return model, tokenizer, use_bf16, device, use_4bit_effective

    if MODEL_MODE == "finetuned":
        if FINETUNED_IS_ADAPTER:
            base_model = AutoModelForCausalLM.from_pretrained(ZERO_SHOT_MODEL_ID, token=hf_token, **model_kwargs)
            model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH, token=hf_token)
            if not use_4bit_effective:
                model.to(device)
            return model, tokenizer, use_bf16, device, use_4bit_effective
        model = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL_PATH, token=hf_token, **model_kwargs)
        if not use_4bit_effective:
            model.to(device)
        return model, tokenizer, use_bf16, device, use_4bit_effective

    raise ValueError("MODEL_MODE must be 'zero_shot' or 'finetuned'")


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


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
    prompt_template = load_prompt_template(PROMPT_PATH)

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
    selected_raw = get_selected_split(raw_dataset, train_pmids, val_pmids, test_pmids)
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
    model, tokenizer, use_bf16, infer_device, use_4bit_effective = load_prediction_model(hf_token=hf_token)
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

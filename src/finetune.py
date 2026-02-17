import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import wandb
from config import (
    EVAL_STEPS,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOGGING_STEPS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_SEQ_LENGTH,
    MAX_STEPS,
    NUM_TRAIN_EPOCHS,
    PER_DEVICE_EVAL_BATCH_SIZE,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    SAVE_STEPS,
    USE_4BIT,
    WEIGHT_DECAY,
)
from datasets import Dataset, load_dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from utils import (
    build_run_config,
    ensure_dirs,
    explode_qa_pairs_with_prompt,
    format_example,
    get_required_env,
    load_prompt_template,
    resolve_private_repo_id,
    setup_logging,
    split_pmids,
    add_nearest_context
)

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
# DATASET_ID = "Jerjes/pubmed_summary_qa_w_encodings"
DATASET_ID = '/Users/jerjesaguirrechavez/Desktop/UCSD/Voytek Lab/NeuroTextAdaptation/data/filtered_dataset_with_vectors' #"Jerjes/pubmed_summary_qa_w_encodings"
# If True, DATASET_ID is interpreted as a Hugging Face dataset repo id.
# If False, DATASET_ID is interpreted as a local dataset path for load_from_disk.
LOAD_FROM_HF = False
BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Reproducibility seed used for PMID shuffle/splits and trainer seed.
RANDOM_SEED = 42
# Dataset split ratios (train, validation, test), must sum to 1.0.
SPLIT_RATIOS = (0.70, 0.15, 0.15)

# Default cap for unique PMIDs to keep runs manageable during development.
MAX_PMIDS_DEFAULT = 1000
# Explicit PMID cap override. Set to None to use MAX_PMIDS_DEFAULT.
# Set this to a positive integer to control training subset size directly.
MAX_PMIDS = None

# Toggle whether retrieval context (nearest papers) is used in prompts.
WITH_CONTEXT = False

# Human-readable run name used for output folder and Hub repo slug.
CONTEXT_TAG = "withctx" if WITH_CONTEXT else "noctx"
RUN_NAME = f"smolllm2-pubmed-qa-{CONTEXT_TAG}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
# Root directory where run artifacts/checkpoints are written.
OUTPUT_ROOT = Path("outputs")
OUTPUT_DIR = OUTPUT_ROOT / RUN_NAME
LOG_PATH = OUTPUT_DIR / "finetune.log"
RUN_CONFIG_PATH = OUTPUT_DIR / "run_config.json"
SPLIT_STATS_PATH = OUTPUT_DIR / "split_stats.json"

# Whether to push model artifacts to Hugging Face Hub.
PUSH_TO_HUB = True
# Strategy for creating hub repo id; currently derived from RUN_NAME.
HUB_REPO_ID_STRATEGY = "derive_from_run_name"
# Make Hub repo private when created.
HUB_PRIVATE_REPO = True
# Require W&B login/API key before training starts.
WANDB_REQUIRED = True
WANDB_PROJECT = "neurotextadaptation-sft"
# Prompt template paths (with/without retrieval context).
PROMPT_WITH_CONTEXT_PATH = Path("src/prompts/qa_context_prompt.txt")
PROMPT_NO_CONTEXT_PATH = Path("src/prompts/qa_no_context_prompt.txt")
# Number of nearest papers to inject as context.
NUM_CONTEXT_PAPERS = 3


def log_pipeline_event(log, stage: str, stage_idx: int, **metrics: float) -> None:
    log.info("Pipeline stage %d: %s", stage_idx, stage)
    if metrics:
        log.info("Stage metrics (%s): %s", stage, metrics)
    if wandb.run is not None:
        payload: dict[str, float] = {"pipeline/stage_index": float(stage_idx)}
        payload.update({f"pipeline/{k}": v for k, v in metrics.items()})
        wandb.log(payload)
        wandb.run.summary["pipeline/last_stage"] = stage


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    global MAX_PMIDS, MAX_STEPS

    parser = argparse.ArgumentParser(description="LoRA SFT for pubmed_summary_qa")
    parser.add_argument("--max_pmids", type=int, default=MAX_PMIDS, help="Override number of PMIDs (None uses dev default).")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS, help="Override max training steps.")
    args_ns = parser.parse_args()

    if args_ns.max_pmids is not None:
        MAX_PMIDS = args_ns.max_pmids
    if args_ns.max_steps is not None:
        MAX_STEPS = args_ns.max_steps

    ensure_dirs(OUTPUT_DIR)
    log = setup_logging(LOG_PATH)
    prompt_path = PROMPT_WITH_CONTEXT_PATH if WITH_CONTEXT else PROMPT_NO_CONTEXT_PATH
    prompt_template = load_prompt_template(prompt_path, require_context=WITH_CONTEXT)

    # ------------------------------------------------------------
    # Environment checks
    # ------------------------------------------------------------
    if torch.cuda.is_available():
        train_device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        train_device = torch.device("mps")
    else:
        train_device = torch.device("cpu")

    hf_token = get_required_env("HF_TOKEN")
    if WANDB_REQUIRED:
        get_required_env("WANDB_API_KEY")

    # ------------------------------------------------------------
    # Save run configuration
    # ------------------------------------------------------------
    run_config = build_run_config(
        dataset_id=DATASET_ID,
        base_model_id=BASE_MODEL_ID,
        random_seed=RANDOM_SEED,
        split_ratios=SPLIT_RATIOS,
        max_pmids_default=MAX_PMIDS_DEFAULT,
        max_pmids=MAX_PMIDS,
        push_to_hub=PUSH_TO_HUB,
        hub_repo_id_strategy=HUB_REPO_ID_STRATEGY,
        hub_private_repo=HUB_PRIVATE_REPO,
        wandb_required=WANDB_REQUIRED,
        wandb_project=WANDB_PROJECT,
        run_name=RUN_NAME,
        output_dir=OUTPUT_DIR,
        training_config={
            "num_train_epochs": NUM_TRAIN_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
            "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "max_length": MAX_SEQ_LENGTH,
            "logging_steps": LOGGING_STEPS,
            "save_steps": SAVE_STEPS,
            "eval_steps": EVAL_STEPS,
            "max_steps": MAX_STEPS,
            "use_4bit": USE_4BIT,
            "num_context_papers": NUM_CONTEXT_PAPERS,
        },
        lora_config={
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": LORA_TARGET_MODULES,
        },
    )
    run_config["load_from_hf"] = LOAD_FROM_HF
    run_config["with_context"] = WITH_CONTEXT
    run_config["prompt_path"] = str(prompt_path)
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
    log.info("Loaded dataset rows: %d", len(raw_dataset))
    log.info("Unique PMIDs found: %d", len(unique_pmids))
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

    log.info("PMID split sizes | train=%d val=%d test=%d", len(train_pmids), len(val_pmids), len(test_pmids))
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
    train_raw = raw_dataset.filter(lambda x: int(x["pmid"]) in train_pmids, desc="Filtering train split")
    val_raw = raw_dataset.filter(lambda x: int(x["pmid"]) in val_pmids, desc="Filtering val split")
    test_raw = raw_dataset.filter(lambda x: int(x["pmid"]) in test_pmids, desc="Filtering test split")
    stage_idx += 1
    log_pipeline_event(
        log,
        "row_split_filtered",
        stage_idx,
        train_rows=float(len(train_raw)),
        val_rows=float(len(val_raw)),
        test_rows=float(len(test_raw)),
        filter_seconds=round(time.perf_counter() - t0, 3),
    )

    # ------------------------------------------------------------
    # Build retrieval context from nearest neuro vectors
    # ------------------------------------------------------------
    if WITH_CONTEXT:
        t0 = time.perf_counter()
        train_raw = add_nearest_context(train_raw, top_k=NUM_CONTEXT_PAPERS, log=log)
        val_raw = add_nearest_context(val_raw, top_k=NUM_CONTEXT_PAPERS, log=log)
        test_raw = add_nearest_context(test_raw, top_k=NUM_CONTEXT_PAPERS, log=log)
        stage_idx += 1
        log_pipeline_event(
            log,
            "retrieval_context_ready",
            stage_idx,
            context_k=float(NUM_CONTEXT_PAPERS),
            train_rows_with_context=float(len(train_raw)),
            val_rows_with_context=float(len(val_raw)),
            test_rows_with_context=float(len(test_raw)),
            retrieval_seconds=round(time.perf_counter() - t0, 3),
        )
    else:
        stage_idx += 1
        log_pipeline_event(
            log,
            "retrieval_context_skipped",
            stage_idx,
            with_context=0.0,
            train_rows=float(len(train_raw)),
            val_rows=float(len(val_raw)),
            test_rows=float(len(test_raw)),
        )

    # ------------------------------------------------------------
    # Convert rows into SFT examples
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    train_sft = explode_qa_pairs_with_prompt(train_raw, prompt_template=prompt_template)
    val_sft = explode_qa_pairs_with_prompt(val_raw, prompt_template=prompt_template)
    test_sft = explode_qa_pairs_with_prompt(test_raw, prompt_template=prompt_template)
    stage_idx += 1
    log_pipeline_event(
        log,
        "sft_examples_built",
        stage_idx,
        train_sft_examples=float(len(train_sft)),
        val_sft_examples=float(len(val_sft)),
        test_sft_examples=float(len(test_sft)),
        build_seconds=round(time.perf_counter() - t0, 3),
    )

    split_stats = {
        "seed": RANDOM_SEED,
        "pmid_counts": {
            "train": len(train_pmids),
            "val": len(val_pmids),
            "test": len(test_pmids),
        },
        "row_counts": {
            "train": len(train_raw),
            "val": len(val_raw),
            "test": len(test_raw),
        },
        "sft_example_counts": {
            "train": len(train_sft),
            "val": len(val_sft),
            "test": len(test_sft),
        },
    }
    SPLIT_STATS_PATH.write_text(json.dumps(split_stats, indent=2), encoding="utf-8")
    log.info("Split stats saved to %s", SPLIT_STATS_PATH)
    log.info("SFT sample counts | train=%d val=%d test=%d", len(train_sft), len(val_sft), len(test_sft))
    if wandb.run is not None:
        wandb.run.summary["data/train_sft_examples"] = len(train_sft)
        wandb.run.summary["data/val_sft_examples"] = len(val_sft)
        wandb.run.summary["data/test_sft_examples"] = len(test_sft)

    # ------------------------------------------------------------
    # Load tokenizer and model
    # ------------------------------------------------------------
    log.info("Loading tokenizer/model from %s", BASE_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_bf16 = bool(train_device.type == "cuda" and torch.cuda.is_bf16_supported())
    if train_device.type == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if use_bf16 else torch.float16

    use_4bit_effective = bool(USE_4BIT and train_device.type == "cuda")
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if use_4bit_effective:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, token=hf_token, **model_kwargs)
    if not use_4bit_effective:
        model.to(train_device)
    stage_idx += 1
    log_pipeline_event(
        log,
        "model_and_tokenizer_loaded",
        stage_idx,
        bf16_enabled=float(int(use_bf16)),
        use_4bit=float(int(use_4bit_effective)),
    )
    log.info("Training device: %s | effective_4bit=%s", train_device, use_4bit_effective)

    # ------------------------------------------------------------
    # Prepare text fields for TRL SFT
    # ------------------------------------------------------------
    train_sft = train_sft.map(
        lambda ex: {"text": format_example(ex, tokenizer)},
        desc="Formatting train chats",
    )
    val_sft = val_sft.map(
        lambda ex: {"text": format_example(ex, tokenizer)},
        desc="Formatting val chats",
    )
    test_sft = test_sft.map(
        lambda ex: {"text": format_example(ex, tokenizer)},
        desc="Formatting test chats",
    )
    stage_idx += 1
    log_pipeline_event(log, "chat_templates_formatted", stage_idx)

    # ------------------------------------------------------------
    # LoRA + trainer setup
    # ------------------------------------------------------------
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
    )

    repo_id = resolve_private_repo_id(hf_token, RUN_NAME)
    run_config["resolved_repo_id"] = repo_id
    RUN_CONFIG_PATH.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    log.info("Resolved private Hub repo: %s", repo_id)
    if wandb.run is not None:
        wandb.config.update({"resolved_repo_id": repo_id}, allow_val_change=True)
    stage_idx += 1
    log_pipeline_event(log, "hub_repo_resolved", stage_idx)

    sft_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        run_name=RUN_NAME,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_length=MAX_SEQ_LENGTH,
        max_steps=MAX_STEPS,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        bf16=bool(train_device.type == "cuda" and use_bf16),
        fp16=bool(train_device.type == "cuda" and not use_bf16),
        seed=RANDOM_SEED,
        report_to=["wandb"],
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=repo_id,
        hub_private_repo=HUB_PRIVATE_REPO,
        hub_token=hf_token,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_sft,
        eval_dataset=val_sft,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    stage_idx += 1
    log_pipeline_event(log, "trainer_initialized", stage_idx)

    # ------------------------------------------------------------
    # Train, evaluate, save, and push
    # ------------------------------------------------------------
    log.info("Starting training")
    stage_idx += 1
    log_pipeline_event(log, "training_started", stage_idx)
    trainer.train()
    log.info("Training finished")
    stage_idx += 1
    log_pipeline_event(log, "training_finished", stage_idx)

    val_metrics = trainer.evaluate()
    prepared_test_sft = trainer._prepare_dataset(
        dataset=test_sft,
        processing_class=tokenizer,
        args=sft_args,
        packing=sft_args.packing,
        formatting_func=None,
        dataset_name="test",
    )
    test_metrics = trainer.evaluate(eval_dataset=prepared_test_sft)
    log.info("Validation metrics: %s", val_metrics)
    log.info("Test metrics: %s", test_metrics)
    if wandb.run is not None:
        wandb.log({f"eval/val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))})
        wandb.log({f"eval/test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))})
    stage_idx += 1
    log_pipeline_event(log, "evaluation_finished", stage_idx)

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    if PUSH_TO_HUB:
        trainer.push_to_hub()
    stage_idx += 1
    log_pipeline_event(log, "artifacts_saved", stage_idx, push_to_hub=float(int(PUSH_TO_HUB)))

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------
    del trainer
    del model
    torch.cuda.empty_cache()
    stage_idx += 1
    log_pipeline_event(log, "cleanup_finished", stage_idx)
    log.info("Finished fine-tuning run")
    wandb.finish()


if __name__ == "__main__":
    main()

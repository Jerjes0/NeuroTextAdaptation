import logging
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import HfApi, create_repo
from peft import PeftModel
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def ensure_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def setup_logging(log_path: Path, logger_name: str = "pubmed-sft") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger(logger_name)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def qa_non_empty(question: Any, answer: Any) -> bool:
    q = "" if question is None else str(question).strip()
    a = "" if answer is None else str(answer).strip()
    return bool(q and a)


def build_run_config(
    *,
    dataset_id: str,
    base_model_id: str,
    random_seed: int,
    split_ratios: tuple[float, float, float],
    max_pmids_default: int,
    max_pmids: int | None,
    push_to_hub: bool,
    hub_repo_id_strategy: str,
    hub_private_repo: bool,
    wandb_required: bool,
    wandb_project: str,
    run_name: str,
    output_dir: Path,
    training_config: dict[str, Any],
    lora_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "base_model_id": base_model_id,
        "random_seed": random_seed,
        "split_ratios": {
            "train": split_ratios[0],
            "val": split_ratios[1],
            "test": split_ratios[2],
        },
        "max_pmids_default": max_pmids_default,
        "max_pmids_effective": max_pmids if max_pmids is not None else max_pmids_default,
        "push_to_hub": push_to_hub,
        "hub_repo_id_strategy": hub_repo_id_strategy,
        "hub_private_repo": hub_private_repo,
        "wandb_required": wandb_required,
        "wandb_project": wandb_project,
        "run_name": run_name,
        "output_dir": str(output_dir),
        "training": training_config,
        "lora": lora_config,
    }


def split_pmids(
    unique_pmids: list[int],
    seed: int,
    split_ratios: tuple[float, float, float],
    max_pmids: int | None,
    max_pmids_default: int,
) -> tuple[set[int], set[int], set[int]]:
    train_ratio, val_ratio, test_ratio = split_ratios
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("SPLIT_RATIOS must sum to 1.0")

    rng = random.Random(seed)
    shuffled = unique_pmids[:]
    rng.shuffle(shuffled)

    max_pmids_effective = max_pmids if max_pmids is not None else max_pmids_default
    selected = shuffled[: max_pmids_effective if max_pmids_effective > 0 else len(shuffled)]

    n_total = len(selected)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_pmids = set(selected[:n_train])
    val_pmids = set(selected[n_train : n_train + n_val])
    test_pmids = set(selected[n_train + n_val : n_train + n_val + n_test])
    return train_pmids, val_pmids, test_pmids


def explode_qa_pairs(dataset: Dataset) -> Dataset:
    default_template = (
        "You are a biomedical assistant.\n\n"
        "Context papers:\n{context}\n\n"
        "Question:\n{question}"
    )
    return explode_qa_pairs_with_prompt(dataset, prompt_template=default_template)


def load_prompt_template(prompt_path: Path, require_context: bool = True) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    template = prompt_path.read_text(encoding="utf-8").strip()
    if "{question}" not in template:
        raise ValueError("Prompt template must contain the {question} placeholder.")
    if require_context and "{context}" not in template:
        raise ValueError("Prompt template must contain the {context} placeholder.")
    return template


def explode_qa_pairs_with_prompt(dataset: Dataset, prompt_template: str) -> Dataset:
    def _explode_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        out = {
            "pmid": [],
            "doi": [],
            "title": [],
            "summary": [],
            "question": [],
            "answer": [],
            "context_block": [],
            "messages": [],
        }
        batch_size = len(batch["pmid"])

        for i in range(batch_size):
            context_block = str(batch.get("context_block", [""] * batch_size)[i] or "").strip()
            for idx in (1, 2, 3):
                q = batch.get(f"question_{idx}", [None] * batch_size)[i]
                a = batch.get(f"answer_{idx}", [None] * batch_size)[i]
                if not qa_non_empty(q, a):
                    continue
                question = str(q).strip()
                answer = str(a).strip()
                user_prompt = prompt_template.format(question=question, context=context_block)
                out["pmid"].append(batch["pmid"][i])
                out["doi"].append(batch.get("doi", [None] * batch_size)[i])
                out["title"].append(batch.get("title", [None] * batch_size)[i])
                out["summary"].append(batch.get("summary", [None] * batch_size)[i])
                out["question"].append(question)
                out["answer"].append(answer)
                out["context_block"].append(context_block)
                out["messages"].append(
                    [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": answer},
                    ]
                )
        return out

    return dataset.map(
        _explode_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Exploding question/answer pairs",
    )


def resolve_private_repo_id(hf_token: str, run_name: str) -> str:
    api = HfApi(token=hf_token)
    user = api.whoami()["name"]
    slug = slugify(run_name)
    repo_id = f"{user}/{slug}"
    create_repo(repo_id=repo_id, private=True, exist_ok=True, token=hf_token)
    return repo_id


def format_example(example: dict[str, Any], tokenizer: Any) -> str:
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

def add_nearest_context(dataset: Dataset, top_k: int, log) -> Dataset:
    if len(dataset) == 0:
        return dataset
    if len(dataset) == 1:
        return dataset.add_column("context_block", [""])

    vectors = np.asarray(dataset["neuro_vector"], dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("Column 'neuro_vector' must be a 2D array-like feature.")

    n_neighbors = min(top_k + 1, len(dataset))
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors, n_jobs=-1)
    nn.fit(vectors)
    _, neighbor_indices = nn.kneighbors(vectors, return_distance=True)

    titles = dataset["title"]
    summaries = dataset["summary"]
    context_blocks: list[str] = []

    for row_idx, neighbors in enumerate(neighbor_indices):
        selected = [int(j) for j in neighbors if int(j) != row_idx][:top_k]
        context_lines = []
        for i, neighbor_idx in enumerate(selected, start=1):
            ctx_title = str(titles[neighbor_idx] or "").strip()
            ctx_summary = str(summaries[neighbor_idx] or "").strip()
            context_lines.append(
                f"Paper {i}\nTitle: {ctx_title}\nAbstract: {ctx_summary}"
            )
        context_blocks.append("\n\n".join(context_lines))

    log.info("Attached nearest-neighbor context blocks for %d rows", len(context_blocks))
    return dataset.add_column("context_block", context_blocks)


def get_selected_split(raw_dataset: Dataset, predict_split: str, train_pmids: set[int], val_pmids: set[int], test_pmids: set[int]) -> Dataset:
    if predict_split == "train":
        return raw_dataset.filter(lambda x: int(x["pmid"]) in train_pmids, desc="Filtering train split")
    if predict_split == "val":
        return raw_dataset.filter(lambda x: int(x["pmid"]) in val_pmids, desc="Filtering val split")
    if predict_split == "test":
        return raw_dataset.filter(lambda x: int(x["pmid"]) in test_pmids, desc="Filtering test split")
    raise ValueError("PREDICT_SPLIT must be one of: train, val, test")


def load_prediction_model(
    *,
    hf_token: str,
    model_mode: str,
    zero_shot_model_id: str,
    finetuned_model_path: str,
    finetuned_is_adapter: bool,
    use_4bit: bool,
) -> tuple[Any, Any, bool, torch.device, bool]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(zero_shot_model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_bf16 = bool(device.type == "cuda" and torch.cuda.is_bf16_supported())
    if device.type == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if use_bf16 else torch.float16

    use_4bit_effective = bool(use_4bit and device.type == "cuda")
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if use_4bit_effective:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    if model_mode == "zero_shot":
        model = AutoModelForCausalLM.from_pretrained(zero_shot_model_id, token=hf_token, **model_kwargs)
        if not use_4bit_effective:
            model.to(device)
        return model, tokenizer, use_bf16, device, use_4bit_effective

    if model_mode == "finetuned":
        if finetuned_is_adapter:
            base_model = AutoModelForCausalLM.from_pretrained(zero_shot_model_id, token=hf_token, **model_kwargs)
            model = PeftModel.from_pretrained(base_model, finetuned_model_path, token=hf_token)
            if not use_4bit_effective:
                model.to(device)
            return model, tokenizer, use_bf16, device, use_4bit_effective
        model = AutoModelForCausalLM.from_pretrained(finetuned_model_path, token=hf_token, **model_kwargs)
        if not use_4bit_effective:
            model.to(device)
        return model, tokenizer, use_bf16, device, use_4bit_effective

    raise ValueError("MODEL_MODE must be 'zero_shot' or 'finetuned'")


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())

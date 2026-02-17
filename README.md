# NeuroTextAdaptation

## Fine-tuning (`src/finetune.py`)

### Prerequisites
- CUDA-enabled GPU (script fails fast if CUDA is not available).
- Hugging Face token with permissions to create/push private repos.
- Weights & Biases API key.

### Setup
```bash
cd "NeuroTextAdaptation"
source neuro_adaptation/bin/activate
pip install -r requirements.txt
```

### Required environment variables
```bash
export HF_TOKEN="your_hf_token"
export WANDB_API_KEY="your_wandb_api_key"
```

Optional:
```bash
export WANDB_PROJECT="neurotextadaptation-sft"
```

### Run examples
Smoke test:
```bash
python src/finetune.py --max_pmids 2000 --max_steps 50
```

Larger run:
```bash
python src/finetune.py --max_pmids 10000
```

### Notes
- Prompt template is loaded from `src/prompts/qa_context_prompt.txt`.
- Artifacts are written to `outputs/<RUN_NAME>/`.
- With `PUSH_TO_HUB=True`, the script creates/pushes to a private Hub repo derived from `RUN_NAME`.
- W&B logs include preprocessing and training pipeline stages (including retrieval-context completion).

## Prediction (`src/predict.py`)

### Zero-shot setup
In `src/predict.py`, set:
- `MODEL_MODE = "zero_shot"`
- `ZERO_SHOT_MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"` (or your preferred base model)
- `DATASET_ID` to your dataset repo.

### Run examples
Smoke test:
```bash
python src/predict.py --max_pmids 2000 --max_predict_examples 200
```

Larger run:
```bash
python src/predict.py --max_pmids 10000
```

### Prediction outputs
- Real-time CSV: `outputs/<RUN_NAME>/predictions.csv`
  - File is created at start and one row is appended/flushed per prediction.
  - Columns: `title`, `abstract`, `question`, `answer`, `context`, `predicted_answer` (plus ids/metadata).
- Metrics JSON: `outputs/<RUN_NAME>/predict_metrics.json`

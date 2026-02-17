# ------------------------------------------------------------
# Training config
# ------------------------------------------------------------
# Number of full passes through training data.
NUM_TRAIN_EPOCHS = 1.0
# Optimizer learning rate.
LEARNING_RATE = 2e-4
# AdamW weight decay coefficient.
WEIGHT_DECAY = 0.0
# Micro-batch size per GPU for training.
PER_DEVICE_TRAIN_BATCH_SIZE = 2
# Micro-batch size per GPU for evaluation.
PER_DEVICE_EVAL_BATCH_SIZE = 2
# Number of micro-batches accumulated before each optimizer step.
GRADIENT_ACCUMULATION_STEPS = 8
# Maximum token length after tokenization/truncation.
MAX_SEQ_LENGTH = 1024
# Log metrics every N update steps.
LOGGING_STEPS = 10
# Save checkpoint every N update steps.
SAVE_STEPS = 200
# Run validation every N update steps.
EVAL_STEPS = 200
# Maximum update steps. Set to -1 to use NUM_TRAIN_EPOCHS instead.
MAX_STEPS = -1

# Enable QLoRA-style 4-bit loading for reduced VRAM usage.
USE_4BIT = True

# ------------------------------------------------------------
# LoRA config
# ------------------------------------------------------------
# LoRA rank (adapter bottleneck dimension).
LORA_R = 16
# LoRA alpha scaling factor.
LORA_ALPHA = 32
# LoRA dropout probability.
LORA_DROPOUT = 0.05
# Transformer modules that receive LoRA adapters.
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

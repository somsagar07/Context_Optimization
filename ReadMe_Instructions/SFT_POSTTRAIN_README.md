# SFT Post-Training Guide

## Overview
This script refines your RL-trained models by training on high-quality correct episodes from your training logs. This can improve accuracy by 2-5%.

## Quick Start

### Step 1: Run SFT Post-Training
```bash
python sft_posttrain.py \
    --rl-log logs/training_log_grpo_gsm8k_1766872133.json \
    --rl-model-dir models/grpo_models \
    --epochs 3 \
    --batch-size 32 \
    --min-reward 4.0
```

### Step 2: Evaluate the Refined Models
```bash
python eval.py \
    --structure-model models/sft_posttrained/structure_policy_sft.pt \
    --prompt-model models/sft_posttrained/prompt_policy_sft.pt \
    --episodes 100
```

## Arguments

- `--rl-log`: Path to your RL training log JSON file (required)
- `--rl-model-dir`: Directory containing RL-trained models (required)
- `--epochs`: Number of SFT training epochs (default: 3)
- `--lr`: Learning rate for SFT (default: 1e-4)
- `--batch-size`: Batch size for training (default: 32, larger = faster but more memory)
- `--min-reward`: Minimum reward threshold to filter high-quality episodes (default: 4.0)
- `--output-dir`: Where to save SFT models (default: models/sft_posttrained)
- `--config`: Config to use (default: hierarchical)

## How It Works

1. **Loads correct episodes** from your RL log that have reward >= min_reward
2. **Loads your RL-trained models** (looks for `*_final.pt` files)
3. **Trains both policies** on these correct episodes using cross-entropy loss
4. **Saves refined models** in the same format as RL models

## Tips

- **Lower `--min-reward`** if you don't have enough high-quality episodes
- **Increase `--epochs`** for more refinement (but watch for overfitting)
- **Use `--lr 5e-5`** for more conservative fine-tuning
- **Increase `--batch-size`** (64, 128) for faster training if you have GPU memory
- **Decrease `--batch-size`** (16, 8) if you run out of memory
- The script automatically finds the most recent models if `_final.pt` files don't exist

## Performance

With batching enabled, training is **significantly faster**:
- **Before**: ~1-2 minutes per epoch (batch_size=1)
- **After**: ~10-20 seconds per epoch (batch_size=32)
- Speedup: **5-10x faster** depending on dataset size



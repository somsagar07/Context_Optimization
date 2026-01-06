# Action Masking

## Overview

Action masking reduces the effective action space from **62,208 to 41,904 combinations** (32.6% reduction) by preventing invalid agent2 selections for workflows that don't use it.

## How It Works

### Two-Stage Selection

When `--mask` is enabled, the policy uses a **two-stage selection process**:

1. **Stage 1**: Select workflow (dimension 0)
2. **Stage 2**: Select other dimensions (agent1_tools, agent1_budget, agent2_tools, agent2_budget, answerer_budget) with masking based on the selected workflow

This ensures that:
- **Workflows 0, 1, 5**: agent2_tools and agent2_budget are masked (only index 0 is valid)
- **Workflows 2, 3, 4, 6, 7, 8**: All dimensions are valid

### Masking Rules

| Workflow | Uses Agent2? | Masking |
|----------|--------------|---------|
| 0 (Direct) | No | agent2_tools and agent2_budget masked |
| 1 (Reason+Ans) | No | agent2_tools and agent2_budget masked |
| 2 (Reason+Verify+Ans) | Yes (Verifier) | No masking |
| 3 (Routing) | Yes (Reasoner2) | No masking |
| 4 (Parallel-Sectioning) | Yes (Worker1) | No masking |
| 5 (Parallel-Voting) | No | agent2_tools and agent2_budget masked |
| 6 (Orchestrator-Workers) | Yes (Workers) | No masking |
| 7 (Evaluator-Optimizer) | Yes (Evaluator) | No masking |
| 8 (Autonomous-Agent) | Yes (Iterations 2+) | No masking |

## Usage

**Without masking (default)**:
```bash
python train.py --algorithm ppo --episodes 20000
```

**With masking**:
```bash
python train.py --algorithm ppo --episodes 20000 --mask
```

## Action Space

| Mode | Action Space | Reduction |
|------|--------------|-----------|
| Without masking | 62,208 | - |
| With masking | 41,904 | **32.6%** |

### Per-Workflow Action Space

- **Workflows 0, 1, 5**: 144 valid combinations each (agent2 masked)
- **Workflows 2, 3, 4, 6, 7, 8**: 6,912 valid combinations each (all valid)

## Implementation Details

### Code Changes

1. **`train.py`**: Added `--mask` flag (default: False)
2. **`algorithms/base.py`**: 
   - Added `use_action_masking` parameter to `BaseTrainer`
   - Modified `get_action()` to support two-stage selection
   - Modified `get_log_prob()` methods to reconstruct masks from actions during training
3. **`algorithms/ppo.py`** and **`algorithms/grpo.py`**: Updated to pass masking flags

### Technical Details

- Masking is applied by setting invalid logits to `-inf` before softmax
- During training, masks are reconstructed from the workflow in the action
- Two-stage selection ensures workflow is known before masking other dimensions
- Compatible with both PPO and GRPO algorithms

## Benefits

1. **Faster Training**: Smaller action space means faster exploration
2. **Better Learning**: Agent doesn't waste time on invalid actions
3. **Cleaner Policies**: Policies learn only valid action combinations
4. **Correctness**: Prevents invalid agent2 selections

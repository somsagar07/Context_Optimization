# Context Optimization via Reinforcement Learning

This project uses Reinforcement Learning (RL) to optimize LLM agent configurations for mathematical reasoning tasks. We explore three different RL formulations: **Single-Step**, **Multi-Step**, and **Hierarchical RL**.

---

## Problem Statement

Given a question (e.g., from GSM8K dataset), we want to automatically configure an LLM agent to maximize correctness while minimizing computational cost. The configuration space includes:

1. **Workflow Structure**: Direct answer, Reason+Answer, or Reason+Verify+Answer
2. **Tools**: Calculator, web search, Python executor (binary combinations)
3. **Token Budgets**: Low (64), Mid (256), or High (512) tokens per agent
4. **Prompts**: Sequential selection from a library of prompt "atoms" for each agent

**Total Configuration Space**: ~5,184 structure combinations × variable prompt selections = massive search space

---

## RL Approaches

### 1. Single-Step RL (Baseline)

**Environment**: `GeneralAgentEnv`  
**Training**: `train.py --config single_step`

#### Formulation
- **Action Space**: `MultiDiscrete([3, 8, 3, 8, 3, 3])` = **5,184 combinations**
  - All 6 dimensions chosen simultaneously in one step
- **Observation Space**: Question embedding (768-dim) + question statistics (8-dim)
- **Reward**: Sparse, only at episode end
  ```
  R = correctness × 5.0 - cost_penalties
  ```
- **Episode Length**: 1 step (single decision)

#### RL 
- **Contextual Bandit**: This is effectively a multi-armed bandit where the context (question) changes each episode
- **No Temporal Credit Assignment**: Since it's single-step, `γ=0.0` (no discounting needed)
- **Large Action Space**: 5,184 actions makes exploration difficult

#### Limitations
- ❌ **Credit Assignment Problem**: Can't learn which specific dimension (workflow vs. tools vs. budget) contributed to success
- ❌ **Poor Exploration**: Large action space makes it hard to explore effectively
- ❌ **No Structure Learning**: Can't learn dependencies between choices (e.g., complex questions need verification)

---

### 2. Multi-Step RL (Sequential Decision Making)

**Environment**: `MultiStepAgentEnv`  
**Training**: `train.py --config multi_step`

#### Formulation
- **Action Space**: **Variable per step** (much smaller!)
  - Step 0: Workflow depth [3 actions]
  - Step 1: Reasoner config (tools + budget) [24 actions]
  - Step 2: Verifier config (if needed) [24 actions]
  - Step 3: Answerer budget [3 actions]
- **Episode Length**: 2-4 steps (variable)
- **Reward**: 
  - **Intermediate rewards** (shaping): Small efficiency bonuses at each step
  - **Final reward**: Correctness + penalties at execution

#### RL 
- **Multi-Step MDP**: Proper Markov Decision Process with temporal structure
- **Temporal Credit Assignment**: Uses `γ=0.95` to discount future rewards
- **Monte Carlo Returns**: `R_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ...`
- **Intermediate Rewards**: Shaping rewards guide exploration (e.g., prefer simpler workflows)

#### Benefits
- ✅ **Smaller Action Spaces**: 3-24 actions per step vs. 5,184
- ✅ **Credit Assignment**: Can learn which step contributed most to success
- ✅ **Dependency Learning**: Can learn that complex questions require Step 2 (verifier)
- ✅ **Better Exploration**: Smaller spaces = easier to explore

#### Example Episode
```
Step 0: Choose workflow_depth=2 (Reason+Verify+Answer)
        → Intermediate reward: +0.02 (efficiency bonus for choosing workflow)
        
Step 1: Choose reasoner: tools=[calc, web], budget=High
        → Intermediate reward: +0.01 (efficiency bonus)
        
Step 2: Choose verifier: tools=[calc], budget=Mid
        → Intermediate reward: +0.01
        
Step 3: Choose answerer: budget=Low
        → Execute workflow → Final reward: +4.5 (correct!) or -0.1 (wrong)
```

---

### 3. Hierarchical RL (Dual-Policy / HIRO-style)

**Environment**: `StructureEnv` + `PromptEnv`  
**Training**: `train_dual.py --config hierarchical`

#### Formulation

Uses **two separate policy networks**:

1. **High-Level Manager (Structure Policy)**
   - **Action Space**: `MultiDiscrete([3, 8, 3, 8, 3, 3])` (same as single-step structure)
   - **Decision**: Single-step selection of workflow, tools, budgets
   - **No Execution**: Only selects the structure
   - **Gamma**: `γ=0.0` (immediate reward, no discounting)

2. **Low-Level Worker (Prompt Policy)**
   - **Action Space**: `Discrete(7)` (select prompt atom or DONE)
   - **Decision**: Sequential selection of prompts for each agent (Reasoner → Verifier → Answerer)
   - **Execution**: Actually runs the LLM workflow with selected structure + prompts
   - **Gamma**: `γ=0.95` (discounts across prompt selection steps)

#### RL 

- **Hierarchical RL**: Two-level decision hierarchy (manager selects goal, worker executes)
- **HIRO-style**: High-level policy selects subgoal (structure), low-level policy executes it (prompts)
- **Shared Reward**: Both policies receive the same final reward from execution
- **Dual Policy Gradient**: Both policies updated via PPO from same episode data
- **Credit Assignment**: 
  - Structure policy: Gets credit for choosing good structure
  - Prompt policy: Gets credit for selecting effective prompts within that structure

#### Episode Flow

```
┌─────────────────────────────────────────────────────────┐
│ HIGH-LEVEL: Structure Policy (Single Step)             │
│ - Observes: Question embedding                          │
│ - Action: [workflow=2, r_tools=5, r_budget=2,          │
│           v_tools=1, v_budget=1, a_budget=0]           │
│ - No reward yet                                         │
└─────────────────┬───────────────────────────────────────┘
                  │ Passes structure to worker
                  ▼
┌─────────────────────────────────────────────────────────┐
│ LOW-LEVEL: Prompt Policy (Multi-Step)                  │
│ Step 1: Select reasoner prompt → "Think step-by-step"  │
│ Step 2: Select reasoner prompt → DONE (stop)           │
│ Step 3: Select verifier prompt → "Check calculations"  │
│ Step 4: Select verifier prompt → DONE                  │
│ Step 5: Select answerer prompt → DONE                  │
│ → Execute workflow with structure + prompts            │
│ → Get final reward: R = correctness × 5.0 - penalties  │
└─────────────────────────────────────────────────────────┘
                  │
                  ▼
        Both policies update from R using PPO
```

#### Benefits
- ✅ **Specialization**: Each policy focuses on its domain (structure vs. prompts)
- ✅ **Interpretability**: Structure decisions are clear and interpretable
- ✅ **Efficient Learning**: Smaller action spaces per policy
- ✅ **Separation of Concerns**: Structure policy doesn't need to learn prompt effects

---

## PPO Implementation Details

All approaches use **Proximal Policy Optimization (PPO)** with the following key features:

### Core PPO Components

1. **Clipped Objective**
   ```
   L^CLIP(θ) = E[min(
       r(θ) · A_t,                    // Unclipped
       clip(r(θ), 1-ε, 1+ε) · A_t    // Clipped
   )]
   ```
   Where `r(θ) = π_θ(a|s) / π_θ_old(a|s)` is the importance sampling ratio, and `ε=0.2` is the clipping parameter.

2. **Value Function Baseline**
   - Reduces variance: `A_t = Q(s_t, a_t) - V(s_t)`
   - Value loss: `L^VF = (V_θ(s_t) - R_t)²`

3. **Entropy Bonus**
   - Encourages exploration: `L = L^CLIP - c_e · H(π(·|s_t))`
   - Entropy coefficient: `c_e = 0.05` (adjustable)

4. **Multiple Epochs**
   - Reuse collected data for 4 epochs per batch (sample efficiency)

5. **Gradient Clipping**
   - `||∇L|| ≤ 0.5` prevents exploding gradients

### Hyperparameters

| Parameter | Single-Step | Multi-Step | Hierarchical |
|-----------|-------------|------------|--------------|
| **Gamma (γ)** | 0.0 | 0.95 | Structure: 0.0, Prompt: 0.95 |
| **Batch Size** | 16-32 | 32-64 | 32 (episodes) |
| **Learning Rate** | 1e-4 | 1e-4 | Structure: 5e-4, Prompt: 1e-4 |
| **Entropy Coef** | 0.02 | 0.01 | Structure: 0.03-0.05, Prompt: 0.05 |


---

## Comparison Summary

| Aspect | Single-Step | Multi-Step | Hierarchical |
|--------|-------------|------------|--------------|
| **Action Space** | 5,184 (all at once) | 3-24 (per step) | Structure: 5,184, Prompt: 7 |
| **Episodes/Steps** | 1 step/episode | 2-4 steps/episode | 1 structure + 3-9 prompt steps |
| **Credit Assignment** | ❌ None | ✅ Temporal (γ=0.95) | ✅ Temporal for prompts |
| **Exploration** | ❌ Difficult | ✅ Easier | ✅ Easier |


---

## Training

### Single-Step or Multi-Step
```bash
python train.py --config single_step    # Single-step baseline
python train.py --config multi_step     # Multi-step sequential
```

### Hierarchical (Dual Policy)
```bash
python train_dual.py \
    --episodes 20000 \
    --entropy-coef 0.05 \
    --batch-size 32 \
    --log-every 100
```

### Evaluation
```bash
# Hierarchical models
python eval_dual.py \
    --structure-model models/structure_policy_*.pt \
    --prompt-model models/prompt_policy_*.pt \
    --episodes 30

# Benchmark (hierarchical + random + optional baselines)
python benchmark_h.py \
    --structure-model models/structure_policy_*.pt \
    --prompt-model models/prompt_policy_*.pt
```

---

## References

- **PPO**: [Schulman et al. (2017)](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization
- **HIRO**: [Nachum et al. (2018)](https://arxiv.org/abs/1805.08296) - Hierarchical RL with Off-Policy Correction
- **Multi-Step RL**: Standard MDP with temporal structure (Sutton & Barto)

---

## File Structure

```
.
├── train.py              # Single/Multi-step training
├── train_dual.py         # Hierarchical (dual-policy) training
├── eval_dual.py          # Evaluation script
├── benchmark_h.py        # Comprehensive benchmarking
├── configs/
│   ├── single_step.py    # Single-step config
│   ├── multi_step.py     # Multi-step config
│   └── hierarchical.py   # Hierarchical config
├── env/
│   ├── general_env.py    # Single-step environment
│   ├── multistep_env.py  # Multi-step environment
│   ├── structure_env.py  # High-level structure policy env
│   └── prompt_env.py     # Low-level prompt policy env
└── agents_system/
    └── worker.py         # LLM execution engine
```


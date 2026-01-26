# Policy Transfer Experiments (Table 2)

This directory contains scripts to run policy transfer experiments for Table 2 in the paper.

## Overview

The script `run_transfer_experiments.py` runs all transfer experiments needed for Table 2:
- Computes dataset similarity (DS) via embedding cosine distance
- Evaluates source accuracy (S_T) on training dataset  
- Evaluates zero-shot accuracy (S_N) on new dataset

## Experiments

### Reasoning Domain
- GSM8K → DROP
- GSM8K → MedQA
- DROP → GSM8K

### Tool Use Domain
- HotpotQA → GAIA
- GAIA → HotpotQA
- HotpotQA → MedQA

## Prerequisites

1. **Pre-trained models**: Models must be trained on source datasets and saved in:
   ```
   models/ppo/{dataset_name}/structure_final.pt
   models/ppo/{dataset_name}/prompt_final.pt
   ```

2. **Precomputed embeddings**: Dataset embeddings must be precomputed:
   ```bash
   python scripts/precompute_embeddings.py --datasets gsm8k drop medqa hotpotqa gaia
   ```

## Usage

### Run All Experiments

```bash
# Using OpenRouter API (recommended for speed)
python scripts/exp_2/run_transfer_experiments.py \
    --all \
    --api \
    --api-model "google/gemini-2.5-flash-lite" \
    --workers 8 \
    --episodes 50

# Using HuggingFace models (slower)
python scripts/exp_2/run_transfer_experiments.py \
    --all \
    --hf-model "Qwen/Qwen2.5-7B-Instruct" \
    --episodes 50
```

### Run Single Experiment

```bash
# GSM8K → DROP transfer
python scripts/exp_2/run_transfer_experiments.py \
    --source gsm8k \
    --target drop \
    --source-structure models/ppo/gsm8k/structure_final.pt \
    --source-prompt models/ppo/gsm8k/prompt_final.pt \
    --api \
    --api-model "google/gemini-2.5-flash-lite" \
    --workers 8 \
    --episodes 50
```

## Output

The script generates:
1. **JSON results file**: `transfer_results/transfer_experiments_{timestamp}.json`
   - Contains all metrics (DS, S_T, S_N) for each experiment
   
2. **LaTeX table**: Printed to console in LaTeX format ready to copy into `experiment.tex`

## Example Output

```
SUMMARY TABLE (for LaTeX)
======================================================================

\begin{table}[t]
    \centering
    \small
    \caption{In-domain policy transfer across datasets...}
    \label{tab:in_domain_transfer}
    ...
    \begin{tabular}{l | l | c | c | c}
        \toprule
        \textbf{Capability} 
        & \textbf{Train $\rightarrow$ New} 
        & \textbf{DS} 
        & \textbf{$S_T$} 
        & \textbf{$S_N$} \\
        \midrule
        \multirow{3}{*}{Reasoning} 
            & GSM8K $\rightarrow$ DROP       & 0.68 & 88.5\% & 51.2\% \\
            ...
    \end{tabular}
\end{table}
```

## Notes

- **Dataset similarity (DS)**: Computed as cosine similarity between average embeddings of source and target datasets
- **Source accuracy (S_T)**: Accuracy of policy on its training dataset
- **Target accuracy (S_N)**: Zero-shot accuracy when policy is evaluated on target dataset
- **Episodes**: Use `--episodes all` to evaluate on full test set (slower but more accurate)
- **Parallel workers**: Only works with `--api` mode. Set `--workers 1` for HuggingFace models

## Troubleshooting

1. **Models not found**: Ensure models are trained and saved in correct location
2. **Embeddings not found**: Run `scripts/precompute_embeddings.py` first
3. **API errors**: Check OpenRouter API key is set in environment
4. **Memory issues**: Reduce `--workers` or use HuggingFace models with `--workers 1`


# Embedding Selection Ablation Study

This directory contains experiments to determine the best embedding model for RL-based context optimization.

## Overview

We test multiple embedding models across 4 different experiments to evaluate:
1. **Clustering Quality** - Do similar questions cluster together?
2. **Classification** - Can embeddings distinguish problem types?
3. **Complexity Ranking** - Do embeddings capture problem difficulty?
4. **Decision Prediction** - Can embeddings predict optimal RL decisions?

## Quick Start

```bash
cd abalation/embedder_selection
python run_all_experiments.py
```

This will:
- Test all embedders defined in `config.py`
- Test each embedder in **BOTH modes**:
  1. **Projected** to same dimension (512D) - fair comparison, RL-ready
  2. **Native** dimensions - no information loss, intrinsic quality
- Run all 4 experiments for each mode
- Generate comparison tables and summary for both modes
- Save results to `embedder_selection_results/`

## Configuration

Edit `config.py` to:
- Change which embedders to test (`EMBEDDERS_TO_TEST`)
- Adjust target embedding dimension (`TARGET_EMBEDDING_DIM`)
- Modify experiment settings
- Set number of samples per dataset

## Experiments

### Experiment 1: Clustering
Tests how well embeddings group similar questions together using K-means clustering.

**Metrics:**
- Silhouette Score (higher = better)
- Adjusted Rand Index vs ground truth (higher = better)
- Intra/Inter-cluster similarity

### Experiment 2: Classification
Tests if embeddings can distinguish problem types (dataset and tool requirements).

**Metrics:**
- Classification accuracy (dataset and tool)
- F1 scores
- Cross-validation scores

### Experiment 3: Complexity Ranking
Tests if embeddings capture problem complexity/difficulty.

**Metrics:**
- Spearman correlation with complexity scores
- R² score for regression
- Ranking accuracy

### Experiment 4: Decision Prediction
Tests if embeddings can predict optimal RL structure decisions.

**Metrics:**
- Workflow prediction accuracy
- Tool selection accuracy
- Budget prediction accuracy
- Combined accuracy

## Output

Results are saved to `embedder_selection_results/`:
- `comparison_table.csv` - Full comparison table (both modes)
- `comparison_table.xlsx` - Excel format (both modes)
- `comparison_table_projected.csv` - Projected mode only
- `comparison_table_native.csv` - Native mode only
- `all_results.json` - Detailed JSON results
- `summary.txt` - Human-readable summary
- `{embedder}_projected_results.json` - Individual embedder results (projected)
- `{embedder}_native_results.json` - Individual embedder results (native)

## Interpreting Results

The **Overall Score** is a normalized average of key metrics:
- Clustering ARI
- Classification accuracy
- Complexity correlation
- Decision prediction score

Higher overall score = better embedder for RL tasks.

### Projected vs Native Dimensions

- **Projected Mode**: All embedders projected to 512D. Fair comparison, directly usable for RL (fixed observation space). May lose some information.

- **Native Mode**: Each embedder uses its natural dimension. No information loss, shows intrinsic quality. Dimensions vary (384D-768D), making comparison less fair but showing raw capability.

**Recommendation**: 
- Use **projected mode** results for RL training (fixed dimension required)
- Compare with **native mode** to see if projection significantly impacts quality
- If native mode is much better, consider using that embedder's native dimension in RL

## Adding New Embedders

1. Add embedder implementation to `embedders.py` (extend `BaseEmbedder`)
2. Add configuration to `EMBEDDER_CONFIGS` in `embedders.py`
3. Add to `EMBEDDERS_TO_TEST` in `config.py`

## Dependencies

Install required packages:
```bash
pip install sentence-transformers scikit-learn pandas openpyxl torch transformers tqdm
# For CLIP:
pip install git+https://github.com/openai/CLIP.git
```

Note: `tqdm` is optional but recommended for progress bars during embedding.


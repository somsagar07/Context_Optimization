# Atom Generation Model Ablation Study

An ablation study to determine the optimal language model for generating dataset-specific prompt atoms. The study evaluates multiple models across different families and sizes using diversity and quality metrics.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Pipeline](#pipeline)
- [Metrics](#metrics)
- [Usage](#usage)
- [Results](#results)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Overview

This study generates prompt atoms for multiple model-dataset combinations and evaluates them using:

1. **Diversity Metrics**: Measures how varied and unique the generated atoms are
2. **Quality Metrics**: Evaluates coherence, specificity, and clarity using GPT-5

The final rankings help identify which models produce the best prompt atoms for different datasets.

## Setup

### Prerequisites

- Python 3.8+
- OpenRouter API key
- Required Python packages (see Installation)

### Installation

1. Navigate to the atom generation directory:
```bash
cd abalation/atom_generation
```

2. Install required dependencies:
```bash
pip install sentence-transformers scikit-learn pandas numpy
```

3. Set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

Or create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your-api-key-here
```

## Pipeline

The ablation study consists of 3 main steps:

| Step | Description | Output |
|------|-------------|--------|
| 1. **Atom Generation** | Generate prompt atoms for all model-dataset combinations | `results/generated_atoms/` |
| 2. **Metric Evaluation** | Compute diversity and quality metrics | `results/metrics/` |
| 3. **Result Aggregation** | Generate rankings and comparison tables | `results/final_rankings/` |

## Metrics

### Diversity Metrics (40% weight)

Diversity metrics measure how varied and unique the generated atoms are:

| Metric | Description | Range | Method |
|--------|-------------|-------|--------|
| **Uniqueness** | How different each atom is from others | 0-1 | `1 - mean(pairwise_cosine_similarity)` using MetaCLIP-H14 embeddings |
| **Strategy Coverage** | How many strategy types are covered | 0-1 | Ratio of unique strategies to total possible strategies (analytical, creative, pedagogical, critical, expert_persona, constraint_focused) |
| **Semantic Diversity** | How semantically spread out atoms are | -1 to 1 | Silhouette score from K-means clustering on MetaCLIP-H14 embeddings |

### Quality Metrics (60% weight)

Quality metrics evaluate the effectiveness and correctness of generated atoms:

| Metric | Description | Range | Method |
|--------|-------------|-------|--------|
| **Coherence** | How well-formed and logical the instruction is | 1-10 | GPT-5 rates each atom on coherence and logical structure |
| **Specificity** | How dataset-specific (vs generic) the atom is | 0-1 | Cosine distance from base atoms using MetaCLIP-H14 embeddings |
| **Clarity** | How clear and actionable the instruction is | 1-10 | GPT-5 rates each atom on clarity and actionability |

### Combined Score

The final combined score is calculated as:

```
Combined = 0.4 × Diversity + 0.6 × Quality
```

Where:
- **Diversity** = average(uniqueness, strategy_coverage, semantic_diversity)
- **Quality** = average(coherence/10, specificity, clarity/10)

All metrics are normalized to 0-1 range before combination.

## Usage

### Full Pipeline

Run the complete ablation study from scratch:

```bash
python run_all_experiments.py
```

This will:
1. Generate atoms for all models and datasets
2. Compute all metrics
3. Generate rankings

### Atoms Already Generated

If you already have atoms generated, skip generation:

```bash
python run_all_experiments.py --skip-generation
```

This will:
1. Skip atom generation (use existing atoms)
2. Compute metrics
3. Generate rankings

### Metrics Already Computed

If you already have metrics, skip both:

```bash
python run_all_experiments.py --skip-generation --skip-metrics
```

This will only aggregate results and generate rankings.

### Only Compute Metrics

To only compute metrics without aggregation:

```bash
python run_all_experiments.py --skip-generation --skip-aggregation
```

### Specific Models or Datasets

To run for specific models and/or datasets:

```bash
# Specific models
python run_all_experiments.py --models openai/gpt-4o anthropic/claude-3.5-sonnet

# Specific datasets
python run_all_experiments.py --datasets gsm8k hotpotqa

# Both
python run_all_experiments.py --models openai/gpt-4o --datasets gsm8k
```

### Command-Line Options

```
--skip-generation      Skip atom generation (use existing atoms)
--skip-metrics         Skip metric evaluation (use existing metrics)
--skip-aggregation     Skip result aggregation (use existing rankings)
--results-dir DIR      Custom results directory (default: results)
--models MODEL1 MODEL2 Specific models to test (default: all in config)
--datasets DATASET1    Specific datasets to test (default: all in config)
```

## Results

### Directory Structure

```
results/
├── generated_atoms/
│   ├── {model-id}/
│   │   ├── {dataset}/
│   │   │   ├── atoms.json          # Generated atoms per role
│   │   │   └── metadata.json       # Generation metadata
│   └── generation_summary.json     # Summary of all generations
├── metrics/
│   ├── {model-id}/
│   │   ├── {dataset}/
│   │   │   └── metrics.json        # Metrics per model-dataset
│   └── all_metrics.json            # All metrics aggregated
└── final_rankings/
    ├── all_results.csv             # Complete results table
    ├── ranking_by_diversity.csv    # Ranked by diversity
    ├── ranking_by_quality.csv      # Ranked by quality
    ├── ranking_by_combined.csv     # Ranked by combined score (recommended)
    └── human_evaluation_format.json # For manual review
```

### Key Files

- **`all_results.csv`**: Complete table with all models, datasets, and scores
- **`ranking_by_combined.csv`**: Models ranked by combined score (use this for final analysis)
- **`human_evaluation_format.json`**: Formatted for manual review and evaluation

### Interpreting Rankings

1. **Combined Score**: Best overall model considering both diversity and quality
2. **Diversity**: Models that generate the most varied atoms
3. **Quality**: Models that generate the most coherent, specific, and clear atoms

## Models Tested

The study tests models across different families and sizes:

| Family | Models | Size Range |
|--------|--------|------------|
| OpenAI | GPT-4o Mini, GPT-4o, GPT-4 Turbo | 1.8T params |
| Anthropic | Claude 3.5 Haiku, Claude 3.5 Sonnet | ~70B params |
| Meta | Llama 3.1 8B, Llama 3.1 70B | 8B-70B params |
| Mistral | Mistral Large | Large |
| Google | Gemini 2.5 Pro | ~1.5T params |
| Qwen | Qwen 2.5 7B, Qwen 2.5 72B | 7B-72B params |

See `models.py` for complete model metadata.

## Datasets

The study tests on multiple datasets:

- **gsm8k**: Math word problems
- **hotpotqa**: Multi-hop question answering
- **gaia**: AI safety and alignment
- **medqa**: Medical question answering
- **aime25**: Advanced math competition problems

## Configuration

### Adjusting Scoring Weights

Edit `config.py` to change the relative importance of diversity vs quality:

```python
SCORE_WEIGHTS = {
    "diversity": 0.40,  # 40% weight
    "quality": 0.60,    # 60% weight
}
```

### Changing Evaluator Model

To use a different model for quality evaluation:

```python
METRICS_CONFIG = {
    "quality": {
        "coherence": {"evaluator_model": "openai/gpt-5"},
        "clarity": {"evaluator_model": "openai/gpt-5"},
    }
}
```

### Adding Models or Datasets

Edit `config.py`:

```python
# Add models
API_MODELS_TO_TEST = [
    "openai/gpt-4o",
    "your-model-id-here",
    # ...
]

# Add datasets
DATASETS = ["gsm8k", "hotpotqa", "your-dataset-here"]
```

### Rate Limiting

Adjust API call delays:

```python
API_RATE_LIMIT_DELAY = 1.0  # Seconds between API calls
API_MAX_RETRIES = 2          # Maximum retries for failed calls
API_TIMEOUT = 60             # Timeout in seconds
```

## Troubleshooting

### API Key Issues

**Error**: `401 Unauthorized` or `Invalid API key`

**Solution**:
1. Check that `OPENROUTER_API_KEY` is set correctly:
   ```bash
   echo $OPENROUTER_API_KEY
   ```

2. Verify the API key in OpenRouter dashboard

3. Ensure the API key has sufficient credits/permissions

### Rate Limiting

**Error**: `429 Too Many Requests`

**Solution**:
1. Increase `API_RATE_LIMIT_DELAY` in `config.py`:
   ```python
   API_RATE_LIMIT_DELAY = 2.0  # Increase delay
   ```

2. Run experiments in smaller batches using `--models` and `--datasets`

3. Use `--skip-*` flags to resume from where it stopped

### Memory Issues

**Error**: `MemoryError` or system running out of memory

**Solution**:
1. Process fewer models/datasets at once:
   ```bash
   python run_all_experiments.py --models openai/gpt-4o --datasets gsm8k
   ```

2. Clear intermediate results if needed

3. Close other applications to free up memory

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution**:
1. Install required dependencies:
   ```bash
   pip install sentence-transformers scikit-learn pandas numpy torch
   ```

2. Or use conda environment:
   ```bash
   conda activate CoRL  # or your environment name
   ```

### Model Not Found

**Error**: `404 Client Error: Not Found for url: https://openrouter.ai/api/v1/chat/completions`

**Solution**:
1. Check that the model ID exists on OpenRouter
2. Verify the model ID spelling in `config.py`
3. Check OpenRouter status page for model availability

## Cost Considerations

API costs depend on:
- Number of models tested
- Number of datasets
- Number of atoms generated per combination
- Metric evaluation (quality metrics use GPT-5)

**Estimated costs** (approximate):
- Atom generation: ~$0.01-0.10 per model-dataset combination
- Quality metrics (GPT-5): ~$0.05-0.20 per model-dataset combination
- Total: ~$5-50 depending on models and datasets used

## Citation

If you use this ablation study in your research, please cite:

```bibtex
@software{atom_generation_ablation,
  title = {Atom Generation Model Ablation Study},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-repo}
}
```

## License

[Add your license information here]

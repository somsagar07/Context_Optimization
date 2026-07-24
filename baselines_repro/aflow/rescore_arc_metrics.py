"""Re-score AFlow test predictions with ARC's exact evaluate_correctness metrics.

AFlow scores HotpotQA/DROP with token F1; ARC (Table 1) uses containment /
numeric-normalized matching. This script re-scores the per-problem prediction
CSVs saved by run_test_eval.py using ARC's own loader classes (imported from
the Context_Optimization repo, instantiated without loading HF data), so the
baseline numbers are computed by byte-identical metric code.

Usage: python rescore_arc_metrics.py <csv_path> <dataset: gsm8k|hotpotqa|drop>
Prints AFlow-native mean score and ARC-metric accuracy.
"""
import importlib
import sys
import types

import pandas as pd

# Import ARC's loader modules without executing utils/__init__.py (which pulls
# in stable_baselines3 and other training-only deps).
ARC = "/data/ssagar6/NeurIPS_26/Context_Optimization"
pkg = types.ModuleType("arc_utils")
pkg.__path__ = [f"{ARC}/utils"]
sub = types.ModuleType("arc_utils.data_loader")
sub.__path__ = [f"{ARC}/utils/data_loader"]
sys.modules["arc_utils"] = pkg
sys.modules["arc_utils.data_loader"] = sub

GSM8kDataset = importlib.import_module("arc_utils.data_loader.gsm8k_loader").GSM8kDataset
HotPotQADataset = importlib.import_module("arc_utils.data_loader.hotpot_loader").HotPotQADataset
DROPDataset = importlib.import_module("arc_utils.data_loader.drop_loader").DROPDataset

LOADERS = {
    "gsm8k": GSM8kDataset,
    "hotpotqa": HotPotQADataset,
    "drop": DROPDataset,
}

def main():
    csv_path, dataset = sys.argv[1], sys.argv[2].lower()
    scorer = LOADERS[dataset].__new__(LOADERS[dataset])  # skip __init__ (no HF load)

    df = pd.read_csv(csv_path)
    n = len(df)
    arc_correct = 0.0
    for _, row in df.iterrows():
        pred = str(row["prediction"]) if pd.notna(row["prediction"]) else ""
        truth = str(row["expected_output"]) if pd.notna(row["expected_output"]) else ""
        arc_correct += scorer.evaluate_correctness(pred, truth)

    print(f"dataset={dataset} n={n}")
    print(f"aflow_native_mean_score={df['score'].mean():.5f}")
    print(f"arc_metric_accuracy={arc_correct / n:.5f}")

if __name__ == "__main__":
    main()

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.inference import run_inference

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_label_names():
    with (PROCESSED_DIR / "label_taxonomy.csv").open("r", newline="", encoding="utf-8") as f:
        return [row["label"] for row in csv.DictReader(f)]


def load_split_targets_and_predictions(split_name, label_names):
    # Load ground truth
    split_csv = PROCESSED_DIR / "splits" / f"{split_name}.csv"
    with split_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    y_true = np.array([[int(row.get(f"y_{label}", "0")) for label in label_names] for row in rows])

    # Load predictions
    pred_csv = PROCESSED_DIR / "inference" / "predictions.csv"
    with pred_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    y_pred = np.array([[int(float(row.get(f"pred_{label}", "0"))) for label in label_names] for row in rows])

    return y_true, y_pred


def compute_metrics(split_name, label_names):
    y_true, y_pred = load_split_targets_and_predictions(split_name, label_names)
    
    micro_f1 = f1_score(y_true.ravel(), y_pred.ravel(), average="micro")
    micro_p = precision_score(y_true.ravel(), y_pred.ravel(), average="micro", zero_division=0)
    micro_r = recall_score(y_true.ravel(), y_pred.ravel(), average="micro", zero_division=0)

    per_label_metrics = []
    for i, label in enumerate(label_names):
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        p = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        r = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        per_label_metrics.append((label, f1, p, r))

    return micro_f1, micro_p, micro_r, per_label_metrics


def print_results(split_name, micro_f1, micro_p, micro_r, per_label_metrics):
    print(f"\n{split_name.upper()} RESULTS")
    print(f"Micro F1: {micro_f1:.4f} | Precision: {micro_p:.4f} | Recall: {micro_r:.4f}")
    print(f"{'Label':<30} {'F1':<10} {'Precision':<12} {'Recall':<10}")
    for label, f1, p, r in per_label_metrics:
        print(f"{label:<30} {f1:<10.4f} {p:<12.4f} {r:<10.4f}")


def main():
    label_names = load_label_names()

    print("RUNNING VALIDATION")
    run_inference(
        input_csv=PROCESSED_DIR / "splits" / "val.csv",
        output_csv=PROCESSED_DIR / "inference" / "predictions.csv",
        escalation_csv=PROCESSED_DIR / "inference" / "escalations.csv",
        model_dir=PROJECT_ROOT / "models" / "3.distilbert_multilabel",
        processed_dir=PROCESSED_DIR,
        max_len=256,
        batch_size=16,
    )
    val_micro_f1, val_micro_p, val_micro_r, val_per_label_metrics = compute_metrics("val", label_names)
    print_results("Validation", val_micro_f1, val_micro_p, val_micro_r, val_per_label_metrics)

    print("RUNNING TEST")
    run_inference(
        input_csv=PROCESSED_DIR / "splits" / "test.csv",
        output_csv=PROCESSED_DIR / "inference" / "predictions.csv",
        escalation_csv=PROCESSED_DIR / "inference" / "escalations.csv",
        model_dir=PROJECT_ROOT / "models" / "3.distilbert_multilabel",
        processed_dir=PROCESSED_DIR,
        max_len=256,
        batch_size=16,
    )
    test_micro_f1, test_micro_p, test_micro_r, test_per_label_metrics = compute_metrics("test", label_names)
    print_results("Test", test_micro_f1, test_micro_p, test_micro_r, test_per_label_metrics)

    print("COMPARISON")
    print(f"Micro F1 Delta: {test_micro_f1 - val_micro_f1:+.4f}")
    print(f"{'Label':<30} {'Val F1':<12} {'Test F1':<12} {'Diff':<12}")
    for (label, val_f1, _, _), (_, test_f1, _, _) in zip(val_per_label_metrics, test_per_label_metrics):
        print(f"{label:<30} {val_f1:<12.4f} {test_f1:<12.4f} {test_f1 - val_f1:+.4f}")


if __name__ == "__main__":
    main()

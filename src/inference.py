from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.preprocessing import preprocess_records


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_labels_and_thresholds(processed_dir: Path) -> tuple[list[str], np.ndarray]:
    with (processed_dir / "label_taxonomy.csv").open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        labels = [row["label"] for row in reader]

    threshold_map = {label: 0.5 for label in labels}
    thresholds_path = processed_dir / "label_thresholds.csv"
    if thresholds_path.exists():
        with thresholds_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                label = str(row["label"])
                if label in threshold_map:
                    threshold_map[label] = float(row["threshold"])

    thresholds = np.array([threshold_map[label] for label in labels], dtype=np.float32)
    return labels, thresholds


def _predict_probabilities(
    texts: list[str],
    tokenizer: DistilBertTokenizerFast,
    model: DistilBertForSequenceClassification,
    device: torch.device,
    max_len: int,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    all_probabilities = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {key: value.to(device) for key, value in enc.items()}
            logits = model(**enc).logits
            all_probabilities.append(torch.sigmoid(logits).cpu().numpy())

    return np.vstack(all_probabilities) if all_probabilities else np.empty((0, model.num_labels), dtype=np.float32)


def _format_explanation(labels: list[str], probabilities_row: np.ndarray, predictions_row: np.ndarray) -> str:
    predicted = [labels[i] for i, value in enumerate(predictions_row) if value == 1]
    ranked = sorted(
        ((labels[i], float(probabilities_row[i])) for i in range(len(labels))),
        key=lambda item: item[1],
        reverse=True,
    )

    if predicted:
        top_bits = ", ".join(f"{label}={score:.2f}" for label, score in ranked[:2])
        return f"Predicted {', '.join(predicted)} with top confidences {top_bits}."

    top_bits = ", ".join(f"{label}={score:.2f}" for label, score in ranked[:3])
    return f"No label passed threshold; highest probabilities were {top_bits}."


def run_inference(
    input_csv: Path,
    output_csv: Path,
    escalation_csv: Path,
    model_dir: Path,
    processed_dir: Path,
    max_len: int,
    batch_size: int,
) -> None:
    with input_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if rows and "text_clean" not in rows[0]:
        rows = preprocess_records(rows)
    else:
        rows = [{**row, "text_clean": row.get("text_clean", "") or ""} for row in rows]

    labels, thresholds = _load_labels_and_thresholds(processed_dir)

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
    model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    probabilities = _predict_probabilities(
        texts=[row["text_clean"] for row in rows],
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_len=max_len,
        batch_size=batch_size,
    )

    predictions = (probabilities >= thresholds.reshape(1, -1)).astype(int)
    output = [dict(row) for row in rows]

    for i, label in enumerate(labels):
        for row, probability, prediction in zip(output, probabilities[:, i], predictions[:, i]):
            row[f"prob_{label}"] = float(probability)
            row[f"pred_{label}"] = int(prediction)
    predicted_labels = [
        "|".join(labels[i] for i, value in enumerate(row) if value == 1)
        for row in predictions
    ]

    explanations = [
        _format_explanation(labels, probabilities_row, predictions_row)
        for probabilities_row, predictions_row in zip(probabilities, predictions)
    ]

    escalation_reasons = [
        "" if np.any(predictions_row) else "no_label_above_threshold"
        for predictions_row in predictions
    ]
    low_confidence = [reason != "" for reason in escalation_reasons]

    for row, predicted, explanation, reason, low in zip(
        output,
        predicted_labels,
        explanations,
        escalation_reasons,
        low_confidence,
    ):
        row["predicted_labels"] = predicted
        row["explanation"] = explanation
        row["escalation_reason"] = reason
        row["low_confidence"] = low

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output[0].keys()) if output else []
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)

    escalation = [row for row in output if row["low_confidence"]]
    escalation_csv.parent.mkdir(parents=True, exist_ok=True)
    with escalation_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(escalation)

    print(f"Saved predictions: {output_csv}")
    print(f"Saved escalations: {escalation_csv} ({len(escalation)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local email inference")
    parser.add_argument("--input-csv", default="data/processed/splits/test.csv")
    parser.add_argument("--output-csv", default="data/processed/inference/predictions.csv")
    parser.add_argument("--escalation-csv", default="data/processed/inference/escalations.csv")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-dir", default="models/distilbert_multilabel")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    project_root = _resolve_project_root()

    input_csv = Path(args.input_csv)
    if not input_csv.is_absolute():
        input_csv = project_root / input_csv

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = project_root / output_csv

    escalation_csv = Path(args.escalation_csv)
    if not escalation_csv.is_absolute():
        escalation_csv = project_root / escalation_csv

    processed_dir = Path(args.processed_dir)
    if not processed_dir.is_absolute():
        processed_dir = project_root / processed_dir

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = project_root / model_dir
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    run_inference(
        input_csv=input_csv,
        output_csv=output_csv,
        escalation_csv=escalation_csv,
        model_dir=model_dir,
        processed_dir=processed_dir,
        max_len=args.max_len,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
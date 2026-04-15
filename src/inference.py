from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from preprocessing import preprocess_dataframe


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_model_dir(project_root: Path, model_dir_arg: str | None) -> Path:
    # Use an explicit model path to keep runtime behavior predictable.
    model_dir = Path(model_dir_arg) if model_dir_arg else project_root / "models" / "3.distilbert_multilabel"
    if not model_dir.is_absolute():
        model_dir = project_root / model_dir
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    return model_dir


def _load_labels_and_thresholds(processed_dir: Path) -> tuple[list[str], np.ndarray]:
    # loads label names from taxonomy and thresholds from CSV
    taxonomy_df = pd.read_csv(processed_dir / "label_taxonomy.csv")
    labels = taxonomy_df["label"].astype(str).tolist()

    threshold_map = {label: 0.5 for label in labels}
    thresholds_path = processed_dir / "label_thresholds.csv"
    if thresholds_path.exists():
        threshold_df = pd.read_csv(thresholds_path)
        for _, row in threshold_df.iterrows():
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
    # predicts probabilities for the input texts
    model.eval()
    all_probs = []

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
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    return np.vstack(all_probs) if all_probs else np.empty((0, model.num_labels), dtype=np.float32)


def run_inference(
    input_csv: Path,
    output_csv: Path,
    model_dir: Path,
    processed_dir: Path,
    max_len: int,
    batch_size: int,
) -> None:
    # loads data, model and thresholds then runs inference and saves results
    df = pd.read_csv(input_csv)
    if "text_clean" not in df.columns:
        df = preprocess_dataframe(df)
    else:
        df = df.copy()
        df["text_clean"] = df["text_clean"].fillna("").astype(str)

    labels, thresholds = _load_labels_and_thresholds(processed_dir)

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
    model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    probs = _predict_probabilities(
        texts=df["text_clean"].tolist(),
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_len=max_len,
        batch_size=batch_size,
    )

    preds = (probs >= thresholds.reshape(1, -1)).astype(int)
    output = df.copy()

    for i, label in enumerate(labels):
        output[f"prob_{label}"] = probs[:, i]
        output[f"pred_{label}"] = preds[:, i]

    output["predicted_labels"] = [
        "|".join(labels[i] for i, value in enumerate(row) if value == 1)
        for row in preds
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)

    print(f"Saved predictions: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local email inference")
    parser.add_argument("--input-csv", default="data/raw_batch1/emails.csv")
    parser.add_argument("--output-csv", default="data/processed/inference/predictions.csv")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-dir", default="models/3.distilbert_multilabel")
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

    processed_dir = Path(args.processed_dir)
    if not processed_dir.is_absolute():
        processed_dir = project_root / processed_dir

    model_dir = _resolve_model_dir(project_root, args.model_dir)

    run_inference(
        input_csv=input_csv,
        output_csv=output_csv,
        model_dir=model_dir,
        processed_dir=processed_dir,
        max_len=args.max_len,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
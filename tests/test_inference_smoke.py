from pathlib import Path
import csv


def test_run_inference_smoke(tmp_path):
    from src.inference import run_inference

    project_root = Path(__file__).resolve().parents[1]

    input_csv = tmp_path / "input.csv"
    with input_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "body"])
        writer.writeheader()
        writer.writerow({"subject": "Test subject", "body": "Hello, I need help with my module."})
        writer.writerow({"subject": "Second", "body": "I need a reference for a job application."})

    output_csv = tmp_path / "predictions.csv"
    escalation_csv = tmp_path / "escalations.csv"

    processed_dir = project_root / "data" / "processed"
    model_dir = project_root / "models" / "4.distilbert_multilabel"

    run_inference(
        input_csv=input_csv,
        output_csv=output_csv,
        escalation_csv=escalation_csv,
        model_dir=model_dir,
        processed_dir=processed_dir,
        max_len=64,
        batch_size=2,
    )

    assert output_csv.exists(), "Predictions file was not created"
    assert escalation_csv.exists(), "Escalations file was not created"

    with output_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert "predicted_labels" in rows[0]

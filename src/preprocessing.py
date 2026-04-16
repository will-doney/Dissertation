from __future__ import annotations

import csv
import re
from pathlib import Path

# Step 1 source: copied from notebooks/01_preprocessing.ipynb
PII_PATTERNS = {
    "EMAIL": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
    "PHONE": r"\b(?:\+44\s?7\d{3}|07\d{3})\s?\d{3}\s?\d{3}\b",
    "STUDENT_ID": r"\b(?:w|W)?\d{7,9}\b",
}

SIGN_OFF_PATTERNS = [
    r"(?im)^kind regards,?.*$",
    r"(?im)^regards,?.*$",
    r"(?im)^thanks,?.*$",
    r"(?im)^many thanks,?.*$",
]


def _safe_text(value: object) -> str:
    # converts non-string inputs to empty strings
    if value is None:
        return ""
    if value != value:
        return ""
    return str(value)


def mask_pii(text: object) -> str:
    # masks PII in the text using regex patterns
    masked = _safe_text(text)
    for label, pattern in PII_PATTERNS.items():
        masked = re.sub(pattern, f"[{label}]", masked)
    return masked


def clean_email_text(text: object) -> str:
    # cleaning steps to remove quoted text, URLs, sign-offs, and extra whitespace
    cleaned = _safe_text(text)
    cleaned = re.sub(r"(?is)\n?On .*? wrote:.*$", " ", cleaned)
    cleaned = re.sub(r"(?im)^>.*$", " ", cleaned)
    cleaned = re.sub(r"https?://\S+|www\.\S+", " [URL] ", cleaned)
    for pattern in SIGN_OFF_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_text_clean(subject: object, body: object) -> str:
    # applies full cleaning pipeline to the subject and body
    subject_clean = clean_email_text(mask_pii(subject))
    body_clean = clean_email_text(mask_pii(body))
    return f"{subject_clean} [SEP] {body_clean}".strip()


def preprocess_records(records: list[dict[str, object]]) -> list[dict[str, str]]:
    """Apply the notebook preprocessing pipeline to a list of CSV records."""
    processed: list[dict[str, str]] = []

    for row in records:
        if "subject" not in row or "body" not in row:
            raise ValueError("Missing required columns: subject/body")

        subject = _safe_text(row.get("subject"))
        body = _safe_text(row.get("body"))
        subject_masked = mask_pii(subject)
        body_masked = mask_pii(body)
        subject_clean = clean_email_text(subject_masked)
        body_clean = clean_email_text(body_masked)

        processed.append(
            {
                **{key: _safe_text(value) for key, value in row.items()},
                "subject_masked": subject_masked,
                "body_masked": body_masked,
                "subject_clean": subject_clean,
                "body_clean": body_clean,
                "text_clean": f"{subject_clean} [SEP] {body_clean}".strip(),
            }
        )

    return processed


def preprocess_csv(input_csv: Path, output_csv: Path) -> None:
    # reads the input CSV, applies preprocessing, and saves the output CSV with new columns
    with input_csv.open("r", newline="", encoding="utf-8") as source_file:
        reader = csv.DictReader(source_file)
        records = list(reader)

    out = preprocess_records(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out[0].keys()) if out else []
    with output_csv.open("w", newline="", encoding="utf-8") as target_file:
        writer = csv.DictWriter(target_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out)

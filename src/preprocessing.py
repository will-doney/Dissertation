from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

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
    if pd.isna(value):
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


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # applies preprocessing pipeline and adds new columns for masked and cleaned text
    required_cols = ["subject", "body"]
    missing_cols = [column for column in required_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    out = df.copy()
    out["subject_masked"] = out["subject"].apply(mask_pii)
    out["body_masked"] = out["body"].apply(mask_pii)
    out["subject_clean"] = out["subject_masked"].apply(clean_email_text)
    out["body_clean"] = out["body_masked"].apply(clean_email_text)
    out["text_clean"] = (out["subject_clean"] + " [SEP] " + out["body_clean"]).str.strip()
    return out


def preprocess_csv(input_csv: Path, output_csv: Path) -> None:
    # reads the input CSV, applies preprocessing, and saves the output CSV with new columns
    df = pd.read_csv(input_csv)
    out = preprocess_dataframe(df)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

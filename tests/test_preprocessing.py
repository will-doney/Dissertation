import math
from pathlib import Path

import pytest

from src import preprocessing as pp


def test_safe_text_none_and_nan_and_number():
    assert pp._safe_text(None) == ""
    assert pp._safe_text(float("nan")) == ""
    assert pp._safe_text(123) == "123"


def test_mask_pii_replaces_email_and_student_id():
    text = "Contact me at alice@example.com or my id W21039284."
    masked = pp.mask_pii(text)
    assert "[EMAIL]" in masked
    assert "[STUDENT_ID]" in masked


def test_clean_email_text_removes_quotes_urls_and_signoff():
    raw = (
        "Hello\n\nSee details at https://example.com/page\n\n" 
        "On Tue, someone wrote:\n> quoted line\n\nKind regards,\nBob"
    )
    cleaned = pp.clean_email_text(raw)
    assert "[URL]" in cleaned or "example.com" not in cleaned
    assert "quoted line" not in cleaned
    assert "Kind regards" not in cleaned


def test_preprocess_records_missing_columns_raises():
    with pytest.raises(ValueError):
        pp.preprocess_records([{"subject": "a"}])


def test_preprocess_records_adds_expected_fields():
    recs = [{"subject": "Hi", "body": "Email me alice@example.com"}]
    out = pp.preprocess_records(recs)
    assert isinstance(out, list)
    assert "text_clean" in out[0]
    assert "subject_masked" in out[0]
    assert "body_masked" in out[0]

"""Unit and integration tests for SupportPilot PII redaction (SP-102)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from supportpilot.preprocessing.__main__ import main as cli_main
from supportpilot.preprocessing.redact import (
    _luhn_valid,
    process_file,
    redact_field,
    redact_text,
)


def test_luhn_validation():
    """Verify Luhn checksum calculation for valid and invalid card numbers."""
    assert _luhn_valid("4111111111111111") is True
    assert _luhn_valid("4111111111111112") is False
    assert _luhn_valid("1234567890123") is False
    # Valid 15-digit Amex test card
    assert _luhn_valid("378282246310005") is True


def test_samples_produce_expected_results():
    """Verify each client test case from data/pii_samples.jsonl."""
    samples_path = Path("data/pii_samples.jsonl")
    assert samples_path.exists(), "Sample file data/pii_samples.jsonl must exist"

    results = {}
    with samples_path.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            ticket_id = item["ticket_id"]
            redacted, counts = redact_field(item["body"])
            results[ticket_id] = (redacted, counts)

    # S-1: Email with + tag and multi-part domain
    assert results["S-1"][0] == "Please email me at [EMAIL] about my refund."
    assert results["S-1"][1]["email"] == 1

    # S-2: Two phone numbers (IN and US)
    assert results["S-2"][0] == "Call me on [PHONE] or [PHONE] after 6pm."
    assert results["S-2"][1]["phone"] == 2

    # S-3: Valid card number
    assert results["S-3"][0] == "My card [CARD] was charged twice for order #55231."
    assert results["S-3"][1]["card"] == 1

    # S-4: Card-shaped mistyped card is redacted (fail-closed), while 13-digit tracking number survives
    assert (
        results["S-4"][0] == "Card [CARD] never arrived. Tracking number 1234567890123."
    )
    assert results["S-4"][1]["card"] == 1

    # S-5: Ticket ID and currency amounts remain untouched
    assert (
        results["S-5"][0]
        == "Reference T-1001 from last week, amount 2500.00 INR, 3 items."
    )
    assert sum(results["S-5"][1].values()) == 0

    # S-6: Multilingual Hindi text with email and phone
    assert results["S-6"][0] == "मेरा ईमेल [EMAIL] है और फोन [PHONE] है।"
    assert results["S-6"][1]["email"] == 1
    assert results["S-6"][1]["phone"] == 1

    # S-7: Already redacted text remains identical
    assert results["S-7"][0] == "Already redacted: [EMAIL] and [PHONE] and [CARD]."
    assert sum(results["S-7"][1].values()) == 0

    # S-8: Null body remains None
    assert results["S-8"][0] is None
    assert sum(results["S-8"][1].values()) == 0


def test_idempotence_on_all_sample_lines():
    """Redacting already redacted text must produce identical text and zero counts across all samples."""
    samples_path = Path("data/pii_samples.jsonl")
    with samples_path.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            body = item["body"]
            first_pass, counts1 = redact_field(body)
            second_pass, counts2 = redact_field(first_pass)
            assert second_pass == first_pass
            assert counts2 == {"card": 0, "email": 0, "phone": 0}


def test_card_fail_closed_and_tracking_numbers():
    """Verify card detection fails closed on card-shaped numbers while preserving tracking numbers."""
    # 1. Grouped Luhn-fail number (spaces) -> redacted
    res1 = redact_text("My card 4111 1111 1111 1112 expired.")
    assert res1.text == "My card [CARD] expired."
    assert res1.counts["card"] == 1

    # 2. Grouped Luhn-fail number (dashes, no keyword) -> redacted
    res2 = redact_text("Account code 4111-1111-1111-1112.")
    assert res2.text == "Account code [CARD]."
    assert res2.counts["card"] == 1

    # 3. Ungrouped Luhn-fail number (e.g. tracking number) -> preserved intact
    res3 = redact_text("Package tracking number 1234567890123.")
    assert res3.text == "Package tracking number 1234567890123."
    assert res3.counts["card"] == 0

    # 4. Ungrouped Luhn-pass number -> redacted
    res4 = redact_text("Payment reference 4111111111111111 received.")
    assert res4.text == "Payment reference [CARD] received."
    assert res4.counts["card"] == 1


def test_safe_strings_must_not_change():
    """Dates, IP addresses, and monetary totals must survive unredacted."""
    safe_strings = [
        "Date 2025-01-14 10:30",
        "IP 192.168.1.1",
        "Total 1,234,567.89",
    ]
    for text in safe_strings:
        result = redact_text(text)
        assert result.text == text
        assert result.counts == {"card": 0, "email": 0, "phone": 0}


def test_unicode_phone_numbers():
    """Verify Devanagari and full-width phone numbers are matched and redacted."""
    res_hindi = redact_text("फोन ९८७६५४३२१०")
    assert res_hindi.text == "फोन [PHONE]"
    assert res_hindi.counts["phone"] == 1

    res_fullwidth = redact_text("ph １２３４５６７８９０")
    assert res_fullwidth.text == "ph [PHONE]"
    assert res_fullwidth.counts["phone"] == 1


def test_hostile_input_redos_safety():
    """Ensure regex processing cannot backtrack catastrophically on hostile repeated inputs."""
    # 1.1 MB of repeated digits, spaces, and dashes
    hostile_input = "1234-5678-9012-3456 " * 55_000
    assert len(hostile_input) > 1_000_000

    start_time = time.perf_counter()
    result = redact_text(hostile_input)
    elapsed = time.perf_counter() - start_time

    assert elapsed < 1.0, f"Hostile input took {elapsed:.2f}s, exceeding 1.0s limit"
    # Grouped card blocks fail closed and are redacted
    assert result.counts["card"] == 55_000


def test_email_edge_cases():
    """Test various valid email formats and boundaries."""
    text = "Emails: user.name+tag@sub.domain.org, test@example.io."
    redacted, counts = redact_text(text)
    assert redacted == "Emails: [EMAIL], [EMAIL]."
    assert counts["email"] == 2


def test_phone_edge_cases():
    """Test US and Indian phone number variations."""
    text = "Call +1-800-555-0199 or 9876543210 or (555) 019-2831."
    redacted, counts = redact_text(text)
    assert redacted == "Call [PHONE] or [PHONE] or [PHONE]."
    assert counts["phone"] == 3


def test_null_and_non_string_fields():
    """Verify null handling and type enforcement."""
    val, counts = redact_field(None)
    assert val is None
    assert counts == {"card": 0, "email": 0, "phone": 0}

    with pytest.raises(TypeError, match="must be strings or null"):
        redact_field(12345)


def test_subject_and_body_both_redacted(tmp_path):
    """Ensure both subject and body fields are redacted in ticket records."""
    input_file = tmp_path / "valid.jsonl"
    output_dir = tmp_path / "out"

    record = {
        "ticket_id": "T-100",
        "subject": "Help for user@example.com",
        "body": "Call 9876543210 regarding 4111-1111-1111-1111",
    }
    input_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    report = process_file(input_file, output_dir)
    assert report["total_records"] == 1
    assert report["total_redactions"] == 3
    assert report["counts"]["email"] == 1
    assert report["counts"]["phone"] == 1
    assert report["counts"]["card"] == 1

    redacted_records = [
        json.loads(line)
        for line in (output_dir / "redacted.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .split("\n")
    ]
    assert redacted_records[0]["subject"] == "Help for [EMAIL]"
    assert redacted_records[0]["body"] == "Call [PHONE] regarding [CARD]"


def test_no_pii_in_report_or_outputs(tmp_path):
    """Verify that original PII strings never appear in report.json or logs."""
    input_file = tmp_path / "valid.jsonl"
    output_dir = tmp_path / "out"

    secret_email = "very_secret_email_999@secretcorp.com"
    secret_card = "4111 1111 1111 1111"
    record = {
        "ticket_id": "T-200",
        "subject": secret_email,
        "body": f"My card is {secret_card}",
    }
    input_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    process_file(input_file, output_dir)

    report_content = (output_dir / "redaction_report.json").read_text(encoding="utf-8")
    redacted_content = (output_dir / "redacted.jsonl").read_text(encoding="utf-8")

    assert secret_email not in report_content
    assert secret_card not in report_content
    assert secret_email not in redacted_content
    assert secret_card not in redacted_content


def test_redaction_report_structure_and_sorted_keys(tmp_path):
    """Verify redaction_report.json formatting, sorted keys, and trailing newline."""
    input_file = tmp_path / "valid.jsonl"
    output_dir = tmp_path / "out"

    record = {
        "ticket_id": "T-300",
        "body": "Contact me at dev@example.com",
    }
    input_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    process_file(input_file, output_dir)

    raw_report = (output_dir / "redaction_report.json").read_text(encoding="utf-8")
    assert raw_report.endswith("\n"), "Report must end with a trailing newline"

    report_data = json.loads(raw_report)
    keys = list(report_data.keys())
    assert keys == ["counts", "total_records", "total_redactions"]
    assert list(report_data["counts"].keys()) == ["card", "email", "phone"]
    assert report_data["total_records"] == 1
    assert report_data["total_redactions"] == 1
    assert report_data["counts"]["email"] == 1


def test_cli_deterministic_and_byte_identical(tmp_path):
    """Running the CLI twice on identical inputs produces byte-identical outputs."""
    input_file = tmp_path / "valid.jsonl"
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    record = {
        "ticket_id": "T-400",
        "subject": "Question",
        "body": "Call +91 98765 43210 or email test@example.com with card 4111 1111 1111 1111",
    }
    input_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with patch(
        "sys.argv",
        ["prog", "--input", str(input_file), "--output-dir", str(out1)],
    ):
        code1 = cli_main()
        assert code1 == 0

    with patch(
        "sys.argv",
        ["prog", "--input", str(input_file), "--output-dir", str(out2)],
    ):
        code2 = cli_main()
        assert code2 == 0

    assert (out1 / "redacted.jsonl").read_bytes() == (
        out2 / "redacted.jsonl"
    ).read_bytes()
    assert (out1 / "redaction_report.json").read_bytes() == (
        out2 / "redaction_report.json"
    ).read_bytes()


def test_cli_refuses_input_overwrite(tmp_path):
    """Verify CLI refuses to overwrite its own output files."""
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    conflicting_input = output_dir / "redacted.jsonl"
    conflicting_input.write_text("{}\n", encoding="utf-8")

    with patch(
        "sys.argv",
        ["prog", "--input", str(conflicting_input), "--output-dir", str(output_dir)],
    ):
        code = cli_main()
        assert code == 1


def test_cli_missing_or_invalid_input(tmp_path):
    """Verify CLI returns exit code 1 when input file is missing or a directory."""
    output_dir = tmp_path / "out"
    missing_file = tmp_path / "does_not_exist.jsonl"

    with patch(
        "sys.argv",
        ["prog", "--input", str(missing_file), "--output-dir", str(output_dir)],
    ):
        assert cli_main() == 1

    with patch(
        "sys.argv",
        ["prog", "--input", str(tmp_path), "--output-dir", str(output_dir)],
    ):
        assert cli_main() == 1


def test_crash_resilience_cleans_stale_report(tmp_path):
    """Verify that any pre-existing report is unlinked at start."""
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_report = output_dir / "redaction_report.json"
    stale_report.write_text('{"stale": true}\n', encoding="utf-8")

    input_file = tmp_path / "valid.jsonl"
    input_file.write_text(
        json.dumps({"ticket_id": "T-1", "body": "hello"}) + "\n",
        encoding="utf-8",
    )

    process_file(input_file, output_dir)

    fresh_report = json.loads(stale_report.read_text(encoding="utf-8"))
    assert "stale" not in fresh_report
    assert fresh_report["total_records"] == 1


def test_idempotence_card_placeholder_in_tracking_window():
    """Ensure [CARD] placeholder does not act as a keyword or expand window across clauses."""
    s = "My card 4111 1111 1111 1112 failed. Tracking 1234567890123 is separate."
    pass1, counts1 = redact_text(s)
    pass2, counts2 = redact_text(pass1)
    assert pass1 == "My card [CARD] failed. Tracking 1234567890123 is separate."
    assert pass2 == pass1
    assert counts2 == {"card": 0, "email": 0, "phone": 0}


def test_cli_non_string_field_clean_exit_without_pii(tmp_path, capsys):
    """Verify CLI exits 1 with clean error and no record values or tracebacks when encountering non-string fields."""
    input_file = tmp_path / "bad.jsonl"
    output_dir = tmp_path / "bad_out"
    secret_value = 12345678
    input_file.write_text(
        json.dumps({"ticket_id": "X-1", "body": secret_value}) + "\n",
        encoding="utf-8",
    )

    with patch(
        "sys.argv",
        ["prog", "--input", str(input_file), "--output-dir", str(output_dir)],
    ):
        code = cli_main()
        assert code == 1

    captured = capsys.readouterr()
    assert "Error: PII-redactable fields must be strings or null" in captured.err
    assert str(secret_value) not in captured.err
    assert str(secret_value) not in captured.out
    assert "Traceback" not in captured.err
    assert not (output_dir / "redacted.jsonl").exists()
    assert not (output_dir / "redaction_report.json").exists()

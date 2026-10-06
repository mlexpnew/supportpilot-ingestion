"""PII detection and redaction for SupportPilot ticket text (SP-102)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])"
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@"
    r"[A-Za-z0-9]"
    r"(?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
    r"(?![\w-])"
)

PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(?:"
    r"(?:\+?1[\s.-]?)?"
    r"(?:\(\d{3}\)|\d{3})"
    r"[\s.-]?"
    r"\d{3}"
    r"[\s.-]?"
    r"\d{4}"
    r"|"
    r"(?:\+?91[\s.-]?)?"
    r"[6-9]\d{4}[\s.-]?\d{5}"
    r")"
    r"(?!\d)"
)

# 13 to 19 digits with optional spaces or dashes, ending on a digit.
CARD_PATTERN = re.compile(r"(?<![\d-])" r"\d(?:[ -]?\d){12,18}" r"(?![\d-])")


@dataclass(frozen=True)
class RedactionResult:
    """Result of redacting PII from a text value."""

    text: str
    counts: dict[str, int]

    def __iter__(self) -> Iterator[Any]:
        """Allow unpacking into (text, counts)."""
        yield self.text
        yield self.counts


def _luhn_valid(value: str) -> bool:
    """Return whether a numeric string passes the Luhn checksum."""
    total = 0
    parity = len(value) % 2

    for index, character in enumerate(value):
        digit = ord(character) - ord("0")

        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9

        total += digit

    return total % 10 == 0


def _is_card_shaped(candidate: str) -> bool:
    """Check if candidate digits have standard payment card grouping."""
    parts = re.split(r"[ -]+", candidate)
    if len(parts) == 4 and all(len(p) == 4 for p in parts):
        return True  # 4-4-4-4 (Visa, Mastercard, Discover, etc.)
    if len(parts) == 3 and [len(p) for p in parts] == [4, 6, 5]:
        return True  # 4-6-5 (Amex)
    if len(parts) == 4 and [len(p) for p in parts] == [4, 4, 4, 3]:
        return True  # 15-digit 4-4-4-3
    return False


def _has_card_context(text: str, start: int) -> bool:
    """Check if text immediately preceding candidate mentions card keywords."""
    window = text[max(0, start - 30) : start]
    return bool(re.search(r"(?i)\b(?:card|visa|mastercard|amex)\b", window))


def _redact_cards(text: str) -> tuple[str, int]:
    """Redact payment-card candidates from text, failing closed on card-shaped numbers."""
    count = 0
    pieces: list[str] = []
    last_end = 0

    for match in CARD_PATTERN.finditer(text):
        candidate = match.group(0)
        digits = candidate.replace(" ", "").replace("-", "")

        if not (13 <= len(digits) <= 19):
            continue

        is_luhn = _luhn_valid(digits)
        is_grouped = _is_card_shaped(candidate)
        has_context = _has_card_context(text, match.start())

        # Fail closed: redact if valid Luhn, or formatted in standard card groups,
        # or explicitly preceded by card keywords (e.g. customer typo).
        if is_luhn or is_grouped or has_context:
            pieces.append(text[last_end : match.start()])
            pieces.append("[CARD]")
            last_end = match.end()
            count += 1

    if count == 0:
        return text, 0

    pieces.append(text[last_end:])
    return "".join(pieces), count


def redact_text(text: str) -> RedactionResult:
    """
    Redact supported PII from text and return counts only.

    Supported types:
    - Email: [EMAIL]
    - Phone: [PHONE]
    - Payment card: [CARD] (13-19 digits, Luhn validated)
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    counts = {
        "card": 0,
        "email": 0,
        "phone": 0,
    }

    # Replace email first so digits inside an email are not interpreted
    # independently as phone/card candidates.
    text, email_count = EMAIL_PATTERN.subn("[EMAIL]", text)
    counts["email"] = email_count

    text, phone_count = PHONE_PATTERN.subn("[PHONE]", text)
    counts["phone"] = phone_count

    text, card_count = _redact_cards(text)
    counts["card"] = card_count

    return RedactionResult(text=text, counts=counts)


def redact_field(value: object) -> tuple[object, dict[str, int]]:
    """Redact a ticket field while preserving null values."""
    if value is None:
        return None, {"card": 0, "email": 0, "phone": 0}

    if not isinstance(value, str):
        raise TypeError("PII-redactable fields must be strings or null")

    result = redact_text(value)
    return result.text, result.counts


def process_file(
    input_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Process a JSONL ticket file, redacting PII from subject and body fields.

    Writes redacted.jsonl and redaction_report.json to output_dir.
    """
    resolved_input = input_path.resolve()
    resolved_output_dir = output_dir.resolve()
    conflicting_outputs = {
        (resolved_output_dir / "redacted.jsonl").resolve(),
        (resolved_output_dir / "redaction_report.json").resolve(),
    }
    if resolved_input in conflicting_outputs:
        raise ValueError(
            f"Input file '{input_path}' conflicts with output files in '{output_dir}'."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove any existing report to ensure failed runs don't leave stale reports
    report_path = output_dir / "redaction_report.json"
    report_path.unlink(missing_ok=True)

    redacted_path = output_dir / "redacted.jsonl"

    tmp_redacted_path = output_dir / ".redacted.jsonl.tmp"
    tmp_report_path = output_dir / ".redaction_report.json.tmp"

    total_records = 0
    total_counts = {"card": 0, "email": 0, "phone": 0}

    with (
        input_path.open("r", encoding="utf-8") as in_file,
        tmp_redacted_path.open("w", encoding="utf-8") as out_file,
    ):
        for line in in_file:
            if not line.strip():
                continue

            record = json.loads(line)
            if not isinstance(record, dict):
                continue

            total_records += 1

            if "subject" in record and record["subject"] is not None:
                redacted_subj, subj_counts = redact_field(record["subject"])
                record["subject"] = redacted_subj
                for k, v in subj_counts.items():
                    total_counts[k] += v

            if "body" in record and record["body"] is not None:
                redacted_body, body_counts = redact_field(record["body"])
                record["body"] = redacted_body
                for k, v in body_counts.items():
                    total_counts[k] += v

            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_redactions = sum(total_counts.values())
    report: dict[str, Any] = {
        "counts": dict(sorted(total_counts.items())),
        "total_records": total_records,
        "total_redactions": total_redactions,
    }

    with tmp_report_path.open("w", encoding="utf-8") as report_file:
        json.dump(
            report,
            report_file,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        report_file.write("\n")

    # Atomically promote temporary output files upon complete run
    tmp_redacted_path.replace(redacted_path)
    tmp_report_path.replace(report_path)

    return report

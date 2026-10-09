import codecs
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from supportpilot.ingestion.__main__ import main as cli_main
from supportpilot.ingestion.loader import (
    BlankFileError,
    EmptyFileError,
    PathConflictError,
    UnsupportedEncodingError,
    process_file,
    validate_lines,
)


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            if isinstance(record, str):
                file.write(record + "\n")
            else:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")


def valid_ticket(ticket_id="T-1001"):
    return {
        "ticket_id": ticket_id,
        "customer_id": "C-100",
        "created_at": "2025-01-14T09:15:00+00:00",
        "channel": "email",
        "body": "Payment failed",
        "status": "open",
    }


def test_empty_body_is_rejected(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    ticket_whitespace = valid_ticket("T-1001")
    ticket_whitespace["body"] = "   "

    ticket_empty = valid_ticket("T-1002")
    ticket_empty["body"] = ""

    write_jsonl(input_file, [ticket_whitespace, ticket_empty])

    report = process_file(input_file, output_dir)

    assert report["valid_records"] == 0
    assert report["invalid_records"] == 2

    reject_lines = (
        (output_dir / "rejects.jsonl").read_text(encoding="utf-8").strip().split("\n")
    )
    reject1 = json.loads(reject_lines[0])
    reject2 = json.loads(reject_lines[1])

    assert reject1["field"] == "body"
    assert reject1["error_type"] == "empty_body"
    assert reject2["field"] == "body"
    assert reject2["error_type"] == "empty_body"


def test_non_iso_timestamp_is_rejected(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    ticket = valid_ticket()
    ticket["created_at"] = "14/01/2025 10:05"
    write_jsonl(input_file, [ticket])

    report = process_file(input_file, output_dir)

    assert report["invalid_records"] == 1
    rejects = (output_dir / "rejects.jsonl").read_text(encoding="utf-8")
    reject = json.loads(rejects.strip())
    assert reject["field"] == "created_at"
    assert reject["error_type"] == "invalid_timestamp"
    assert reject["ticket_id"] == "T-1001"


def test_timezone_naive_timestamp_is_rejected(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    ticket = valid_ticket()
    ticket["created_at"] = "2025-01-14T09:15:00"
    write_jsonl(input_file, [ticket])

    report = process_file(input_file, output_dir)

    assert report["invalid_records"] == 1
    rejects = (output_dir / "rejects.jsonl").read_text(encoding="utf-8")
    reject = json.loads(rejects.strip())
    assert reject["field"] == "created_at"
    assert reject["error_type"] == "invalid_timestamp"


def test_unknown_channel_is_rejected(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    ticket = valid_ticket()
    ticket["channel"] = "carrier_pigeon"
    write_jsonl(input_file, [ticket])

    report = process_file(input_file, output_dir)

    assert report["invalid_records"] == 1
    rejects = (output_dir / "rejects.jsonl").read_text(encoding="utf-8")
    reject = json.loads(rejects.strip())
    assert reject["field"] == "channel"
    assert reject["error_type"] == "invalid_channel"
    assert reject["ticket_id"] == "T-1001"


def test_duplicate_ticket_id_is_rejected_with_ticket_id(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    write_jsonl(
        input_file,
        [
            valid_ticket("T-1001"),
            valid_ticket("T-1001"),
        ],
    )

    report = process_file(input_file, output_dir)

    assert report["valid_records"] == 1
    assert report["invalid_records"] == 1

    rejects = (output_dir / "rejects.jsonl").read_text(encoding="utf-8")
    reject = json.loads(rejects.strip())

    assert reject["ticket_id"] == "T-1001"
    assert reject["field"] == "ticket_id"
    assert reject["error_type"] == "duplicate_ticket_id"
    assert "Payment failed" not in rejects


def test_corrupt_json_is_rejected(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    write_jsonl(input_file, ["{this is not valid json"])

    report = process_file(input_file, output_dir)

    assert report["invalid_records"] == 1
    rejects = (output_dir / "rejects.jsonl").read_text(encoding="utf-8")
    reject = json.loads(rejects.strip())
    assert reject["error_type"] == "invalid_json"


def test_non_object_json_is_rejected(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    write_jsonl(input_file, ["[1, 2, 3]"])

    report = process_file(input_file, output_dir)

    assert report["invalid_records"] == 1
    rejects = (output_dir / "rejects.jsonl").read_text(encoding="utf-8")
    reject = json.loads(rejects.strip())
    assert reject["error_type"] == "invalid_record_type"


def test_blank_line_is_counted_separately(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    with input_file.open("w", encoding="utf-8") as file:
        file.write(json.dumps(valid_ticket("T-1001")) + "\n")
        file.write("\n")
        file.write(json.dumps(valid_ticket("T-1002")) + "\n")

    report = process_file(input_file, output_dir)

    assert report["valid_records"] == 2
    assert report["invalid_records"] == 0
    assert report["blank_lines_skipped"] == 1
    assert (
        report["valid_records"]
        + report["invalid_records"]
        + report["blank_lines_skipped"]
        == 3
    )


def test_null_body_is_rejected(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    ticket = valid_ticket()
    ticket["body"] = None
    write_jsonl(input_file, [ticket])

    report = process_file(input_file, output_dir)

    assert report["invalid_records"] == 1
    rejects = (output_dir / "rejects.jsonl").read_text(encoding="utf-8")
    reject = json.loads(rejects.strip())
    assert reject["field"] == "body"
    assert reject["error_type"] == "invalid_type"


def test_invalid_record_does_not_reserve_ticket_id(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    invalid_ticket = valid_ticket("T-2001")
    invalid_ticket["body"] = None

    valid = valid_ticket("T-2001")

    write_jsonl(input_file, [invalid_ticket, valid])

    report = process_file(input_file, output_dir)

    assert report["valid_records"] == 1
    assert report["invalid_records"] == 1


def test_rejects_do_not_contain_ticket_text(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    secret_text = "MY_PRIVATE_CUSTOMER_INFORMATION"
    ticket = valid_ticket()
    ticket["body"] = "   "

    write_jsonl(input_file, [ticket])

    process_file(input_file, output_dir)

    rejects = (output_dir / "rejects.jsonl").read_text(encoding="utf-8")

    assert secret_text not in rejects
    assert "Payment failed" not in rejects


def test_numeric_timestamp_is_rejected(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    ticket = valid_ticket()
    ticket["created_at"] = 1736846100
    write_jsonl(input_file, [ticket])

    report = process_file(input_file, output_dir)

    assert report["valid_records"] == 0
    assert report["invalid_records"] == 1

    rejects = (output_dir / "rejects.jsonl").read_text(encoding="utf-8")
    reject = json.loads(rejects.strip())
    assert reject["field"] == "created_at"
    assert reject["error_type"] == "invalid_timestamp"


@pytest.mark.parametrize(
    "timestamp_val",
    ["1736846100", "1736846100.5", " 1736846100", "+1736846100"],
)
def test_numeric_timestamp_string_is_rejected(tmp_path, timestamp_val):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    ticket = valid_ticket()
    ticket["created_at"] = timestamp_val
    write_jsonl(input_file, [ticket])

    report = process_file(input_file, output_dir)

    assert report["valid_records"] == 0
    assert report["invalid_records"] == 1

    rejects = (output_dir / "rejects.jsonl").read_text(encoding="utf-8")
    reject = json.loads(rejects.strip())
    assert reject["field"] == "created_at"
    assert reject["error_type"] == "invalid_timestamp"


def test_utf8_bom_is_handled(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    content = (
        codecs.BOM_UTF8 + json.dumps(valid_ticket("T-9999")).encode("utf-8") + b"\n"
    )
    input_file.write_bytes(content)

    report = process_file(input_file, output_dir)

    assert report["valid_records"] == 1
    assert report["invalid_records"] == 0


def test_invalid_encoding_is_rejected_without_pii(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    sensitive_bytes = b'{"ticket_id": "T-SECRET", "body": "secret \xff\xfe data"}\n'
    valid_line_1 = json.dumps(valid_ticket("T-1001")).encode("utf-8") + b"\n"
    valid_line_2 = json.dumps(valid_ticket("T-1002")).encode("utf-8") + b"\n"

    input_file.write_bytes(valid_line_1 + sensitive_bytes + valid_line_2)

    report = process_file(input_file, output_dir)

    assert report["total_records"] == 3
    assert report["valid_records"] == 2
    assert report["invalid_records"] == 1
    assert report["rejection_reasons"]["invalid_encoding"] == 1

    rejects = (output_dir / "rejects.jsonl").read_text(encoding="utf-8")
    reject = json.loads(rejects.strip())
    assert reject["line_number"] == 2
    assert reject["error_type"] == "invalid_encoding"
    assert "T-SECRET" not in rejects
    assert "secret" not in rejects


def test_validate_lines_yields_blanks_and_records(tmp_path):
    input_file = tmp_path / "input.jsonl"
    lines = [
        json.dumps(valid_ticket("T-1001")),
        "",
        "{invalid json",
    ]
    input_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    results = list(validate_lines(input_path=input_file))
    assert len(results) == 3

    assert results[0][0] == 1
    assert results[0][1] is not None
    assert results[0][1].ticket_id == "T-1001"
    assert results[0][2] is None

    assert results[1][0] == 2
    assert results[1][1] is None
    assert results[1][2] is None

    assert results[2][0] == 3
    assert results[2][1] is None
    assert results[2][2]["error_type"] == "invalid_json"


def test_empty_string_field_is_rejected_as_empty_field(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    t1 = valid_ticket("T-1")
    t1["ticket_id"] = ""

    t2 = valid_ticket("T-2")
    t2["customer_id"] = "   "

    write_jsonl(input_file, [t1, t2])

    report = process_file(input_file, output_dir)

    assert report["invalid_records"] == 2
    reject_lines = (
        (output_dir / "rejects.jsonl").read_text(encoding="utf-8").strip().split("\n")
    )
    r1 = json.loads(reject_lines[0])
    r2 = json.loads(reject_lines[1])
    assert r1["field"] == "ticket_id"
    assert r1["error_type"] == "empty_field"
    assert r2["field"] == "customer_id"
    assert r2["error_type"] == "empty_field"


def test_wrong_typed_value_is_rejected_as_invalid_type(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    ticket = valid_ticket("T-1")
    ticket["body"] = 12345
    write_jsonl(input_file, [ticket])

    report = process_file(input_file, output_dir)

    assert report["invalid_records"] == 1
    rejects = (output_dir / "rejects.jsonl").read_text(encoding="utf-8")
    reject = json.loads(rejects.strip())
    assert reject["field"] == "body"
    assert reject["error_type"] == "invalid_type"
    assert reject["ticket_id"] == "T-1"


def test_ticket_id_included_on_parsed_rejects_and_capped(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    valid_id_ticket = valid_ticket("T-VALID-ID")
    valid_id_ticket["channel"] = "invalid_chan"

    long_id_ticket = valid_ticket("T-" + "X" * 100)
    long_id_ticket["channel"] = "invalid_chan"

    write_jsonl(input_file, [valid_id_ticket, long_id_ticket])

    report = process_file(input_file, output_dir)

    assert report["invalid_records"] == 2
    reject_lines = (
        (output_dir / "rejects.jsonl").read_text(encoding="utf-8").strip().split("\n")
    )
    r1 = json.loads(reject_lines[0])
    r2 = json.loads(reject_lines[1])

    assert r1["ticket_id"] == "T-VALID-ID"
    assert "ticket_id" not in r2


def test_source_overwrite_is_prevented(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    conflicting_input = output_dir / "valid.jsonl"
    conflicting_input.write_text("{}\n", encoding="utf-8")

    with pytest.raises(PathConflictError, match="conflicts with output files"):
        process_file(conflicting_input, output_dir)


def test_overwrite_refusal_cli_exits_code_2(tmp_path, capsys):
    output_dir = tmp_path / "output_conflict"
    output_dir.mkdir(parents=True, exist_ok=True)
    conflicting_file = output_dir / "valid.jsonl"
    conflicting_file.write_text('{"ticket_id": "T-1"}\n', encoding="utf-8")

    test_args = [
        "prog",
        "--input",
        str(conflicting_file),
        "--output-dir",
        str(output_dir),
    ]
    with patch.object(sys, "argv", test_args):
        code = cli_main()
        assert code == 2

    captured = capsys.readouterr()
    assert "Path conflict:" in captured.err
    summary = json.loads(captured.out.strip().split("\n")[-1])
    assert summary["exit_reason"] == "bad_file"


def test_fail_on_rejects_cli_flag(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    ticket_bad = valid_ticket("T-BAD")
    ticket_bad["channel"] = "unknown_channel"
    ticket_good = valid_ticket("T-GOOD")
    write_jsonl(input_file, [ticket_good, ticket_bad])

    test_args = [
        "prog",
        "--input",
        str(input_file),
        "--output-dir",
        str(output_dir),
        "--fail-on-rejects",
    ]
    with patch.object(sys, "argv", test_args):
        code = cli_main()
        assert code == 3

    # Without flag, exits 0 because total < 20 and valid > 0
    test_args_no_flag = [
        "prog",
        "--input",
        str(input_file),
        "--output-dir",
        str(output_dir),
    ]
    with patch.object(sys, "argv", test_args_no_flag):
        code = cli_main()
        assert code == 0


def test_validation_errors_hide_input():
    from pydantic import ValidationError

    from supportpilot.ingestion.models import Ticket

    try:
        Ticket.model_validate(
            {
                "ticket_id": "T-1",
                "customer_id": "C-1",
                "created_at": "secret_bad_date",
                "channel": "email",
                "body": "secret_customer_data",
                "status": "open",
            }
        )
    except ValidationError as exc:
        assert "secret_bad_date" not in str(exc)
        assert "input_value" not in str(exc)


def test_invalid_status_is_rejected(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    ticket = valid_ticket()
    ticket["status"] = "closed"
    write_jsonl(input_file, [ticket])

    report = process_file(input_file, output_dir)

    assert report["invalid_records"] == 1
    rejects = (output_dir / "rejects.jsonl").read_text(encoding="utf-8")
    reject = json.loads(rejects.strip())
    assert reject["field"] == "status"
    assert reject["error_type"] == "invalid_status"


def test_line_too_long_is_rejected(tmp_path, monkeypatch):
    import supportpilot.ingestion.loader as loader_mod

    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    monkeypatch.setattr(loader_mod, "MAX_LINE_BYTES", 200)

    long_line = json.dumps(valid_ticket("T-1")) + " " * 500 + "\n"
    normal_line = json.dumps(valid_ticket("T-2")) + "\n"

    input_file.write_bytes(long_line.encode("utf-8") + normal_line.encode("utf-8"))

    report = process_file(input_file, output_dir)

    assert report["total_records"] == 2
    assert report["valid_records"] == 1
    assert report["invalid_records"] == 1
    assert report["rejection_reasons"]["line_too_long"] == 1


def test_deeply_nested_json_recursion_error_is_rejected_as_invalid_json(
    tmp_path, monkeypatch
):
    import supportpilot.ingestion.loader as loader_mod

    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    def mock_loads(s):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(loader_mod.json, "loads", mock_loads)
    input_file.write_text('{"a": 1}\n', encoding="utf-8")

    report = process_file(input_file, output_dir)
    monkeypatch.undo()

    assert report["invalid_records"] == 1
    rejects = (output_dir / "rejects.jsonl").read_text(encoding="utf-8")
    reject = json.loads(rejects.strip())
    assert reject["error_type"] == "invalid_json"


def test_utf16_le_bom_processes_correctly(tmp_path):
    input_file = tmp_path / "input_utf16le.jsonl"
    output_dir = tmp_path / "output_utf16le"

    ticket1 = valid_ticket("T-LE-1")
    ticket2 = valid_ticket("T-LE-2")
    content = codecs.BOM_UTF16_LE + (
        json.dumps(ticket1) + "\n" + json.dumps(ticket2) + "\n"
    ).encode("utf-16-le")
    input_file.write_bytes(content)

    report = process_file(input_file, output_dir)
    assert report["total_records"] == 2
    assert report["valid_records"] == 2
    assert report["invalid_records"] == 0

    valid_lines = (
        (output_dir / "valid.jsonl").read_text(encoding="utf-8").strip().split("\n")
    )
    assert len(valid_lines) == 2
    assert json.loads(valid_lines[0])["ticket_id"] == "T-LE-1"
    assert json.loads(valid_lines[1])["ticket_id"] == "T-LE-2"


def test_utf16_be_bom_processes_correctly(tmp_path):
    input_file = tmp_path / "input_utf16be.jsonl"
    output_dir = tmp_path / "output_utf16be"

    ticket1 = valid_ticket("T-BE-1")
    ticket2 = valid_ticket("T-BE-2")
    content = codecs.BOM_UTF16_BE + (
        json.dumps(ticket1) + "\n" + json.dumps(ticket2) + "\n"
    ).encode("utf-16-be")
    input_file.write_bytes(content)

    report = process_file(input_file, output_dir)
    assert report["total_records"] == 2
    assert report["valid_records"] == 2
    assert report["invalid_records"] == 0

    valid_lines = (
        (output_dir / "valid.jsonl").read_text(encoding="utf-8").strip().split("\n")
    )
    assert len(valid_lines) == 2
    assert json.loads(valid_lines[0])["ticket_id"] == "T-BE-1"
    assert json.loads(valid_lines[1])["ticket_id"] == "T-BE-2"


def test_utf16_without_bom_fails_as_file_level_error(tmp_path, capsys):
    input_file = tmp_path / "input_nobom.jsonl"
    output_dir = tmp_path / "output_nobom"

    # Encode without BOM using 'utf-16-le'
    content = (json.dumps(valid_ticket("T-NOBOM")) + "\n").encode("utf-16-le")
    input_file.write_bytes(content)

    # 1. Loader raises UnsupportedEncodingError
    with pytest.raises(
        UnsupportedEncodingError, match="UTF-16 without BOM is not supported"
    ):
        process_file(input_file, output_dir)

    # 2. CLI exits with code 2
    test_args = ["prog", "--input", str(input_file), "--output-dir", str(output_dir)]
    with patch.object(sys, "argv", test_args):
        code = cli_main()
        assert code == 2

    captured = capsys.readouterr()
    summary = json.loads(captured.out.strip().split("\n")[-1])
    assert summary["exit_reason"] == "unsupported_encoding"


def test_mixed_encodings_in_file(tmp_path):
    input_file = tmp_path / "mixed.jsonl"
    output_dir = tmp_path / "output_mixed"

    valid_line_1 = (json.dumps(valid_ticket("T-M1")) + "\n").encode("utf-8")
    invalid_bytes_line = b'{"ticket_id": "T-BAD", "body": "corrupt \x80\xff line"}\n'
    valid_line_2 = (json.dumps(valid_ticket("T-M2")) + "\n").encode("utf-8")

    input_file.write_bytes(valid_line_1 + invalid_bytes_line + valid_line_2)

    report = process_file(input_file, output_dir)
    assert report["total_records"] == 3
    assert report["valid_records"] == 2
    assert report["invalid_records"] == 1
    assert report["rejection_reasons"]["invalid_encoding"] == 1


def test_reject_rate_threshold_exactly_at_boundary(tmp_path):
    input_file = tmp_path / "boundary.jsonl"
    output_dir = tmp_path / "output_boundary"

    # Create exactly 20 records: 10 valid, 10 invalid -> reject rate = 10 / 20 = 0.50
    records = []
    for i in range(10):
        records.append(json.dumps(valid_ticket(f"T-VAL-{i}")))
    for i in range(10):
        bad_ticket = valid_ticket(f"T-INV-{i}")
        bad_ticket["body"] = ""  # empty_body
        records.append(json.dumps(bad_ticket))

    input_file.write_text("\n".join(records) + "\n", encoding="utf-8")

    # Boundary test: reject_rate == max_reject_rate (0.5 == 0.5) must PASS (exit 0)
    test_args = [
        "prog",
        "--input",
        str(input_file),
        "--output-dir",
        str(output_dir),
        "--max-reject-rate",
        "0.5",
    ]
    with patch.object(sys, "argv", test_args):
        code = cli_main()
        assert code == 0

    # Exceed boundary: add 1 more invalid record -> 11 invalid out of 21 (~0.5238 > 0.50)
    bad_ticket_extra = valid_ticket("T-INV-EXTRA")
    bad_ticket_extra["body"] = ""
    records.append(json.dumps(bad_ticket_extra))
    input_file.write_text("\n".join(records) + "\n", encoding="utf-8")

    with patch.object(sys, "argv", test_args):
        code = cli_main()
        assert code == 3


def test_empty_file_exits_code_2(tmp_path, capsys):
    input_file = tmp_path / "empty.jsonl"
    input_file.touch()
    output_dir = tmp_path / "output_empty"

    with pytest.raises(EmptyFileError, match="Input file is empty"):
        process_file(input_file, output_dir)

    test_args = ["prog", "--input", str(input_file), "--output-dir", str(output_dir)]
    with patch.object(sys, "argv", test_args):
        code = cli_main()
        assert code == 2

    captured = capsys.readouterr()
    summary = json.loads(captured.out.strip().split("\n")[-1])
    assert summary["total"] == 0
    assert summary["valid"] == 0
    assert summary["invalid"] == 0
    assert summary["reject_rate"] == 0.0
    assert summary["exit_reason"] == "empty_file"


def test_blank_only_file_exits_code_2(tmp_path, capsys):
    input_file = tmp_path / "blank_only.jsonl"
    input_file.write_text("\n   \n\t  \n\n", encoding="utf-8")
    output_dir = tmp_path / "output_blank"

    with pytest.raises(BlankFileError, match="Input file contains only blank lines"):
        process_file(input_file, output_dir)

    test_args = ["prog", "--input", str(input_file), "--output-dir", str(output_dir)]
    with patch.object(sys, "argv", test_args):
        code = cli_main()
        assert code == 2

    captured = capsys.readouterr()
    summary = json.loads(captured.out.strip().split("\n")[-1])
    assert summary["total"] == 0
    assert summary["valid"] == 0
    assert summary["invalid"] == 0
    assert summary["reject_rate"] == 0.0
    assert summary["exit_reason"] == "blank_only_file"


def test_cli_exit_codes(tmp_path):
    output_dir = tmp_path / "out"

    # Exit 2: non-existent file
    with patch.object(
        sys,
        "argv",
        [
            "prog",
            "--input",
            str(tmp_path / "nonexistent.jsonl"),
            "--output-dir",
            str(output_dir),
        ],
    ):
        assert cli_main() == 2

    # Exit 0: normal processing (<20 records default threshold does not trigger)
    valid_file = tmp_path / "valid.jsonl"
    write_jsonl(valid_file, [valid_ticket("T-1"), valid_ticket("T-2")])
    with patch.object(
        sys,
        "argv",
        ["prog", "--input", str(valid_file), "--output-dir", str(output_dir)],
    ):
        assert cli_main() == 0

    # Exit 3: reject rate too high
    with patch.object(
        sys,
        "argv",
        [
            "prog",
            "--input",
            str(valid_file),
            "--output-dir",
            str(output_dir),
            "--fail-on-rejects",
        ],
    ):
        bad_ticket = valid_ticket("T-BAD")
        bad_ticket["channel"] = "invalid_chan"
        write_jsonl(valid_file, [bad_ticket])
        assert cli_main() == 3

    # Exit 1: crash
    import supportpilot.ingestion.__main__ as main_mod

    with patch.object(
        main_mod, "process_file", side_effect=RuntimeError("unexpected fatal crash")
    ):
        with patch.object(
            sys,
            "argv",
            ["prog", "--input", str(valid_file), "--output-dir", str(output_dir)],
        ):
            assert cli_main() == 1


def test_summary_line_contains_no_pii_or_ids(tmp_path, capsys):
    input_file = tmp_path / "pii_test.jsonl"
    output_dir = tmp_path / "output_pii"

    secret_body = "Super secret credit card number 4111-2222-3333-4444 and secret_user@example.com"
    ticket = valid_ticket("T-SECRET-ID-999")
    ticket["body"] = secret_body
    write_jsonl(input_file, [ticket])

    test_args = ["prog", "--input", str(input_file), "--output-dir", str(output_dir)]
    with patch.object(sys, "argv", test_args):
        code = cli_main()
        assert code == 0

    captured = capsys.readouterr()
    summary_raw = captured.out.strip().split("\n")[-1]
    summary = json.loads(summary_raw)

    assert set(summary.keys()) == {
        "total",
        "valid",
        "invalid",
        "reject_rate",
        "elapsed",
        "exit_reason",
    }
    assert summary["total"] == 1
    assert summary["valid"] == 1
    assert summary["invalid"] == 0
    assert summary["exit_reason"] == "ok"

    # Strict check: verify no PII or identifiers leaked in summary JSON
    assert "T-SECRET-ID-999" not in summary_raw
    assert "secret_user@example.com" not in summary_raw
    assert "4111-2222-3333-4444" not in summary_raw
    assert "Super secret" not in summary_raw


def test_utf8_baseline_regression_byte_identical(tmp_path):
    baseline_dir = Path("/tmp/baseline")
    if not baseline_dir.exists():
        pytest.skip("Baseline directory /tmp/baseline not present")

    output_dir = tmp_path / "baseline_check"
    sample_path = Path("data/sample_tickets.jsonl")

    report = process_file(sample_path, output_dir)
    assert report["total_records"] == 10
    assert report["valid_records"] == 4
    assert report["invalid_records"] == 6

    # Byte-identical checks
    for fname in ["valid.jsonl", "rejects.jsonl", "report.json"]:
        baseline_bytes = (baseline_dir / fname).read_bytes()
        actual_bytes = (output_dir / fname).read_bytes()
        assert actual_bytes == baseline_bytes, f"Mismatch in {fname}"


def test_incident_export_file_processes_correctly(tmp_path):
    incident_path = Path("data/incident_export.jsonl")
    if not incident_path.exists():
        pytest.skip("Incident export file data/incident_export.jsonl not found")

    output_dir = tmp_path / "incident_check"
    report = process_file(incident_path, output_dir)

    assert report["total_records"] == 10
    assert report["valid_records"] == 4
    assert report["invalid_records"] == 6

    baseline_dir = Path("/tmp/baseline")
    if baseline_dir.exists():
        assert (output_dir / "valid.jsonl").read_bytes() == (
            baseline_dir / "valid.jsonl"
        ).read_bytes()
        assert (output_dir / "rejects.jsonl").read_bytes() == (
            baseline_dir / "rejects.jsonl"
        ).read_bytes()
        assert (output_dir / "report.json").read_bytes() == (
            baseline_dir / "report.json"
        ).read_bytes()


def test_small_file_all_invalid_records_fails(tmp_path):
    input_file = tmp_path / "allbad.jsonl"
    output_dir = tmp_path / "output_allbad"

    # Small file (3 lines), all invalid records
    input_file.write_text("x\nx\nx\n", encoding="utf-8")

    test_args = ["prog", "--input", str(input_file), "--output-dir", str(output_dir)]
    with patch.object(sys, "argv", test_args):
        code = cli_main()
        assert code == 3


def test_crash_handler_masks_exception_payload_pii(tmp_path, capsys):
    input_file = tmp_path / "dummy.jsonl"
    output_dir = tmp_path / "dummy_out"
    input_file.write_text('{"a": 1}\n', encoding="utf-8")

    import supportpilot.ingestion.__main__ as main_mod

    secret_pii = "SECRET_PAYLOAD_SSN_987-65-4321_DO_NOT_LEAK"
    with patch.object(
        main_mod, "process_file", side_effect=RuntimeError(f"crash with {secret_pii}")
    ):
        test_args = [
            "prog",
            "--input",
            str(input_file),
            "--output-dir",
            str(output_dir),
        ]
        with patch.object(sys, "argv", test_args):
            code = cli_main()
            assert code == 1

    captured = capsys.readouterr()
    assert secret_pii not in captured.out
    assert secret_pii not in captured.err
    assert "RuntimeError" in captured.err


def test_reject_rate_unrounded_comparison(tmp_path):
    input_file = tmp_path / "unrounded.jsonl"
    output_dir = tmp_path / "output_unrounded"

    # 1001 invalid out of 2000 records = 0.5005 > 0.50
    records = []
    for i in range(999):
        records.append(json.dumps(valid_ticket(f"T-V-{i}")))
    for i in range(1001):
        bad = valid_ticket(f"T-B-{i}")
        bad["body"] = ""
        records.append(json.dumps(bad))
    input_file.write_text("\n".join(records) + "\n", encoding="utf-8")

    test_args = [
        "prog",
        "--input",
        str(input_file),
        "--output-dir",
        str(output_dir),
        "--max-reject-rate",
        "0.5",
    ]
    with patch.object(sys, "argv", test_args):
        code = cli_main()
        assert code == 3

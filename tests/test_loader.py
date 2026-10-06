import codecs
import json
import sys
from unittest.mock import patch

import pytest

from supportpilot.ingestion.__main__ import main as cli_main
from supportpilot.ingestion.loader import process_file, validate_lines


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


def test_numeric_timestamp_string_is_rejected(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    ticket = valid_ticket()
    ticket["created_at"] = "1736846100"
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

    with pytest.raises(ValueError, match="conflicts with output files"):
        process_file(conflicting_input, output_dir)


def test_fail_on_rejects_cli_flag(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    ticket = valid_ticket()
    ticket["channel"] = "unknown_channel"
    write_jsonl(input_file, [ticket])

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
        assert code == 2

    # Without flag, exits 0
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

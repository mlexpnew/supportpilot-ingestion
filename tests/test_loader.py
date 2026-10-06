import json

from supportpilot.ingestion.loader import process_file


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

    ticket = valid_ticket()
    ticket["body"] = "   "
    write_jsonl(input_file, [ticket])

    report = process_file(input_file, output_dir)

    assert report["valid_records"] == 0
    assert report["invalid_records"] == 1


def test_non_iso_timestamp_is_rejected(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    ticket = valid_ticket()
    ticket["created_at"] = "14/01/2025 10:05"
    write_jsonl(input_file, [ticket])

    report = process_file(input_file, output_dir)

    assert report["invalid_records"] == 1


def test_timezone_naive_timestamp_is_rejected(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    ticket = valid_ticket()
    ticket["created_at"] = "2025-01-14T09:15:00"
    write_jsonl(input_file, [ticket])

    report = process_file(input_file, output_dir)

    assert report["invalid_records"] == 1


def test_unknown_channel_is_rejected(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    ticket = valid_ticket()
    ticket["channel"] = "carrier_pigeon"
    write_jsonl(input_file, [ticket])

    report = process_file(input_file, output_dir)

    assert report["invalid_records"] == 1


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


def test_non_object_json_is_rejected(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_dir = tmp_path / "output"

    write_jsonl(input_file, ["[1, 2, 3]"])

    report = process_file(input_file, output_dir)

    assert report["invalid_records"] == 1


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

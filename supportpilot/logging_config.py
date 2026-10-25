"""Structured JSON logging configuration and sensitive data sanitization."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, TextIO

STANDARD_LOG_RECORD_ATTRS: frozenset[str] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "id",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)

SENSITIVE_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "body",
        "ticket_body",
        "payload_body",
        "request_body",
        "x-api-key",
        "x_api_key",
        "api_key",
        "database_url",
        "db_url",
        "db_uri",
        "database_uri",
        "secret",
        "password",
        "token",
        "authorization",
    }
)

# Regex patterns matching database URLs, credentials, API keys, and bodies
RE_DB_URL = re.compile(
    r"\b(?:postgresql|postgres|sqlite|mysql|mariadb|redis|mongodb)"
    r"(?:\+[a-zA-Z0-9_-]+)?:\/\/[^\s\"',;]+",
    re.IGNORECASE,
)
RE_DB_URL_PARAM = re.compile(r"(?i)\b((?:database_url|db_url)\s*[:=]\s*)([^\s,;]+)")
RE_API_KEY_PARAM = re.compile(
    r"(?i)\b((?:x[-_]?api[-_]?key|api[-_]?key)(?!_hash)\s*[:=]\s*)([^\s,;]+)"
)
RE_BEARER_TOKEN = re.compile(r"(?i)\b(bearer\s+)([A-Za-z0-9_\-\.]+)\b")
RE_BODY_PARAM = re.compile(
    r"(?i)\b((?:ticket_)?body\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|.*?(?=(?:\s+[a-zA-Z0-9_-]+[=:]|\s*$)))"
)

MASKED_REPLACEMENT: str = "[REDACTED]"

VALID_LOG_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def mask_sensitive_text(text: str) -> str:
    """Mask credentials, database URLs, API keys, and ticket bodies in a string."""
    if not text:
        return text

    # Mask database connection URLs first (handles full URLs)
    text = RE_DB_URL.sub(MASKED_REPLACEMENT, text)
    # Mask database URL assignments if any remain
    text = RE_DB_URL_PARAM.sub(r"\1" + MASKED_REPLACEMENT, text)
    # Mask API key assignments (preserves api_key_hash)
    text = RE_API_KEY_PARAM.sub(r"\1" + MASKED_REPLACEMENT, text)
    # Mask Bearer tokens
    text = RE_BEARER_TOKEN.sub(r"\1" + MASKED_REPLACEMENT, text)
    # Mask ticket body assignments
    text = RE_BODY_PARAM.sub(r"\1" + MASKED_REPLACEMENT, text)

    return text


def sanitize_value(key: str | None, value: Any) -> Any:
    """Sanitize a value based on its key name and content."""
    normalized_key = key.lower().replace("-", "_") if key is not None else ""

    # Never mask audit hash `api_key_hash`
    if normalized_key == "api_key_hash":
        return value

    # If key name itself indicates a sensitive field, mask immediately
    if normalized_key in SENSITIVE_FIELD_NAMES or key in SENSITIVE_FIELD_NAMES:
        return MASKED_REPLACEMENT

    if isinstance(value, str):
        return mask_sensitive_text(value)
    if isinstance(value, dict):
        return {k: sanitize_value(k, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_value(None, item) for item in value]
    return value


class SensitiveDataFilter(logging.Filter):
    """Filter that drops or masks sensitive data (API keys, DB URLs, bodies)."""

    def __init__(
        self,
        name: str = "",
        drop: bool = False,
        drop_fields: bool = False,
    ) -> None:
        super().__init__(name)
        self.drop = drop
        self.drop_fields = drop_fields

    def filter(self, record: logging.LogRecord) -> bool:
        formatted_msg = record.getMessage()

        has_sensitive_data = (
            bool(RE_DB_URL.search(formatted_msg))
            or bool(RE_DB_URL_PARAM.search(formatted_msg))
            or bool(RE_API_KEY_PARAM.search(formatted_msg))
            or bool(RE_BEARER_TOKEN.search(formatted_msg))
            or bool(RE_BODY_PARAM.search(formatted_msg))
            or any(
                k.lower().replace("-", "_") in SENSITIVE_FIELD_NAMES
                for k in record.__dict__
                if k not in STANDARD_LOG_RECORD_ATTRS and k != "api_key_hash"
            )
        )

        if self.drop and has_sensitive_data:
            return False

        # Mask message
        masked_msg = mask_sensitive_text(formatted_msg)
        record.msg = masked_msg
        record.args = ()

        # Mask or drop extra fields
        if self.drop_fields:
            for k in list(record.__dict__.keys()):
                if k not in STANDARD_LOG_RECORD_ATTRS and k != "api_key_hash":
                    norm_k = k.lower().replace("-", "_")
                    if norm_k in SENSITIVE_FIELD_NAMES or k in SENSITIVE_FIELD_NAMES:
                        delattr(record, k)
        else:
            for k in list(record.__dict__.keys()):
                if k not in STANDARD_LOG_RECORD_ATTRS and not k.startswith("_"):
                    val = getattr(record, k)
                    setattr(record, k, sanitize_value(k, val))

        return True


class JSONFormatter(logging.Formatter):
    """Formatter that outputs one JSON object per log record line."""

    def format(self, record: logging.LogRecord) -> str:
        # Format message safely
        message = record.getMessage()
        masked_message = mask_sensitive_text(message)

        # UTC timestamp in ISO 8601 format
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()

        log_data: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": masked_message,
        }

        # Include extra attributes
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in STANDARD_LOG_RECORD_ATTRS:
                continue
            if key not in log_data:
                log_data[key] = sanitize_value(key, value)

        # Format exception information if present
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            log_data["exception"] = mask_sensitive_text(record.exc_text)

        if record.stack_info:
            log_data["stack_info"] = mask_sensitive_text(
                self.formatStack(record.stack_info)
            )

        return json.dumps(log_data, default=str)


def get_log_level() -> int:
    """Read LOG_LEVEL from environment variable with fail-fast validation.

    Defaults to INFO if not set. Raises ValueError for invalid levels.
    """
    raw_level = os.getenv("LOG_LEVEL")
    if raw_level is None:
        return logging.INFO

    normalized = raw_level.strip().upper()
    if not normalized or normalized not in VALID_LOG_LEVELS:
        valid_options = ", ".join(VALID_LOG_LEVELS.keys())
        raise ValueError(
            f"Invalid LOG_LEVEL '{raw_level}'. Expected one of: {valid_options}."
        )

    return VALID_LOG_LEVELS[normalized]


_CONFIGURED_STREAM: TextIO = sys.stdout


def setup_logging(stream: TextIO | None = None) -> None:
    """Configure structured JSON logging once at startup."""
    global _CONFIGURED_STREAM

    if stream is not None:
        _CONFIGURED_STREAM = stream

    level = get_log_level()
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clean up previously configured supportpilot JSON handlers
    for handler in list(root_logger.handlers):
        if getattr(handler, "_supportpilot_json_handler", False):
            root_logger.removeHandler(handler)

    handler = logging.StreamHandler(_CONFIGURED_STREAM)
    handler.setLevel(level)
    handler.setFormatter(JSONFormatter())
    handler._supportpilot_json_handler = True  # type: ignore[attr-defined]

    # Attach filter to handler (covers records from root and all child loggers)
    sens_filter = SensitiveDataFilter()
    handler.addFilter(sens_filter)

    root_logger.addHandler(handler)

"""Pydantic models and normalization helpers for SupportPilot Feedback API (SP-110)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from supportpilot.triage import LABELS

MAX_ID_LENGTH: int = 64


def normalize_ticket_id(v: object, max_length: int = MAX_ID_LENGTH) -> str:
    """Normalize and validate ticket_id: strip whitespace and enforce 1-64 char limits."""
    if not isinstance(v, str):
        raise TypeError("ticket_id must be a string")
    s = v.strip()
    if not s:
        raise ValueError("ticket_id cannot be empty")
    if len(s) > max_length:
        raise ValueError(f"ticket_id exceeds maximum length of {max_length} characters")
    return s


def normalize_agent_id(v: object, max_length: int = MAX_ID_LENGTH) -> str:
    """Normalize and validate agent_id: strip whitespace and enforce 1-64 char limits."""
    if not isinstance(v, str):
        raise TypeError("agent_id must be a string")
    s = v.strip()
    if not s:
        raise ValueError("agent_id cannot be empty")
    if len(s) > max_length:
        raise ValueError(f"agent_id exceeds maximum length of {max_length} characters")
    return s


class FeedbackRequest(BaseModel):
    """Request payload for POST /feedback."""

    ticket_id: str = Field(
        min_length=1,
        max_length=MAX_ID_LENGTH,
        description="Unique ticket identifier (1-64 characters)",
    )
    correct_label: str = Field(
        min_length=1, description="Correct category label verified by agent"
    )
    agent_id: str = Field(
        min_length=1,
        max_length=MAX_ID_LENGTH,
        description="Opaque identifier of the human agent (e.g. AG-102, 1-64 characters)",
    )

    @field_validator("ticket_id", mode="before")
    @classmethod
    def validate_ticket_id(cls, v: object) -> str:
        return normalize_ticket_id(v)

    @field_validator("correct_label")
    @classmethod
    def validate_correct_label(cls, v: str) -> str:
        s = v.strip().lower()
        if s not in LABELS:
            raise ValueError(f"Invalid correct_label '{v}'. Must be one of {LABELS}")
        return s

    @field_validator("agent_id", mode="before")
    @classmethod
    def validate_agent_id(cls, v: object) -> str:
        return normalize_agent_id(v)


class FeedbackResponse(BaseModel):
    """Response payload for POST /feedback."""

    ticket_id: str
    correct_label: str
    predicted_label: str
    agreed: bool

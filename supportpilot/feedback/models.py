"""Pydantic models for SupportPilot Feedback API (SP-110 Part B)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from supportpilot.triage import LABELS


class FeedbackRequest(BaseModel):
    """Request payload for POST /feedback."""

    ticket_id: str = Field(min_length=1, description="Unique ticket identifier")
    correct_label: str = Field(
        min_length=1, description="Correct category label verified by agent"
    )
    agent_id: str = Field(min_length=1, description="Identifier of the human agent")

    @field_validator("ticket_id")
    @classmethod
    def validate_ticket_id(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("ticket_id cannot be empty")
        return s

    @field_validator("correct_label")
    @classmethod
    def validate_correct_label(cls, v: str) -> str:
        s = v.strip().lower()
        if s not in LABELS:
            raise ValueError(f"Invalid correct_label '{v}'. Must be one of {LABELS}")
        return s

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("agent_id cannot be empty")
        return s


class FeedbackResponse(BaseModel):
    """Response payload for POST /feedback."""

    ticket_id: str
    correct_label: str
    predicted_label: str
    agreed: bool

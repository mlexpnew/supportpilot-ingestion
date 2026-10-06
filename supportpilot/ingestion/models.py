from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Channel(str, Enum):
    EMAIL = "email"
    CHAT = "chat"
    PHONE = "phone"
    SOCIAL = "social"


class Status(str, Enum):
    OPEN = "open"
    PENDING = "pending"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class Ticket(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ticket_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    created_at: datetime
    channel: Channel
    subject: str | None = None
    body: str = Field(min_length=1)
    language: str = "en"
    status: Status

    @field_validator("ticket_id", "customer_id", "language")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("body")
    @classmethod
    def validate_body(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timezone information is required")
        return value

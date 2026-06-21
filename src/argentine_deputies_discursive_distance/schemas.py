"""Validated data schemas used by the project pipeline."""

from datetime import date, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DiscoveryStatus(StrEnum):
    """Availability of a direct transcript PDF in the source index."""

    PDF_AVAILABLE = "pdf_available"
    NO_PDF_LINK = "no_pdf_link"


class EventStatus(StrEnum):
    """Best status inferred conservatively from the official index."""

    HELD = "held"
    NOT_HELD = "not_held"
    FAILED = "failed"
    UNKNOWN = "unknown"


class SessionCategory(StrEnum):
    """Broad institutional category of a listed parliamentary event."""

    LEGISLATIVE_DEBATE = "legislative_debate"
    INFORMATIVE = "informative"
    PREPARATORY = "preparatory"
    ASSEMBLY = "assembly"
    EXPRESSIONS_IN_MINORITY = "expressions_in_minority"
    HOMAGE = "homage"
    BUDGET_PRESENTATION = "budget_presentation"
    OTHER = "other"


class SessionTerm(StrEnum):
    """Parliamentary term under which a sitting was convened."""

    ORDINARY = "ordinary"
    EXTRAORDINARY = "extraordinary"
    EXTENSION = "extension"
    UNKNOWN = "unknown"


class SessionManifestRecord(BaseModel):
    """One entry listed in the official Diario de Sesiones index."""

    model_config = ConfigDict(frozen=True)

    source_record_id: str = Field(pattern=r"^[0-9a-f]{20}$")
    source_entry_position: int = Field(ge=1)
    period_entry_position: int = Field(ge=1)

    chamber: Literal["deputies"] = "deputies"
    period: int = Field(ge=1)

    meeting_number: int | None = Field(default=None, ge=1)
    session_number: int | None = Field(default=None, ge=1)
    session_date: date

    entry_text_raw: str
    title_raw: str
    title_normalized: str

    session_category: SessionCategory
    session_term: SessionTerm
    event_status: EventStatus
    discovery_status: DiscoveryStatus

    is_special: bool
    is_remote: bool
    is_continuation: bool
    is_joint: bool

    source_page_url: str
    viewer_url: str | None
    pdf_url: str | None

    source_snapshot_at: datetime
    in_candidate_window: bool

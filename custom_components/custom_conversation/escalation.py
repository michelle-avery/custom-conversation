"""Deterministic escalation classification for slower conversation backends."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .const import ESCALATION_SENTINEL

DEFAULT_DOCTRINE = (
    "remember; what did i tell you; what did i say; last time; you said; "
    "remind me; add to the calendar; tomorrow at; recipe; meal plan; "
    "document; invoice; find documents; search documents; pantry; stock; "
    "how is the house; give me the report; house report; what happened overnight"
)
DEFAULT_DENYLIST = "gate; lock; alarm; unlock; open; arm; disarm; siren"
DEFAULT_ACKNOWLEDGEMENT = "Let me look into that and get back to you."


@dataclass(frozen=True)
class Classification:
    """Result of the deterministic pre-routing check."""

    escalate: bool
    security_denied: bool = False


def _contains_phrase(text: str, phrase: str) -> bool:
    """Match a phrase without matching inside another word."""
    phrase = phrase.strip()
    if not phrase:
        return False
    return bool(
        re.search(
            rf"(?<!\w){re.escape(phrase)}(?!\w)",
            text,
            flags=re.IGNORECASE,
        )
    )


def classify_transcript(
    text: str,
    doctrine: str = DEFAULT_DOCTRINE,
    denylist: str = DEFAULT_DENYLIST,
) -> Classification:
    """Apply the denylist before checking the escalation doctrine."""
    if any(_contains_phrase(text, phrase) for phrase in denylist.split(";")):
        return Classification(False, security_denied=True)
    return Classification(
        any(_contains_phrase(text, phrase) for phrase in doctrine.split(";"))
    )


def acknowledgement_for(acknowledgement: str = DEFAULT_ACKNOWLEDGEMENT) -> str:
    """Return the configured acknowledgement, falling back when empty."""
    return acknowledgement.strip() or DEFAULT_ACKNOWLEDGEMENT


def find_unexposed_entity(
    messages: list[object], exposed_entities: set[str]
) -> str | None:
    """Return an entity ID in a tool call that is outside the exposed set."""
    for message in messages:
        for match in re.finditer(
            r"\b[a-z_][\w]*\.[a-z0-9_]+\b", str(message), re.IGNORECASE
        ):
            entity_id = match.group(0)
            if entity_id not in exposed_entities:
                return entity_id
    return None


def post_check_reason(
    speech: str, messages: list[object], exposed_entities: set[str]
) -> str | None:
    """Classify a primary response that should be handed to escalation."""
    if ESCALATION_SENTINEL in speech.upper():
        return "primary-sentinel"
    if entity_id := find_unexposed_entity(messages, exposed_entities):
        return f"primary-unexposed-entity:{entity_id}"
    return None

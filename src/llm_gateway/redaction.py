"""Best-effort PII redaction for logs.

Regexes catch the common, well-structured cases (emails, card numbers, phone
numbers, government IDs, secrets). They do not catch names or free-text
addresses - that needs an NER model (e.g. Presidio) and is listed as a next step.
"""

from __future__ import annotations

import re
from typing import Any


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return "[REDACTED_CARD]" if 13 <= len(digits) <= 19 and _luhn_ok(digits) else match.group(0)


def _phone(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return "[REDACTED_PHONE]" if 10 <= len(digits) <= 15 else match.group(0)


# Order matters: more specific patterns run first.
_PATTERNS: list[tuple[re.Pattern[str], Any]] = [
    (re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_\-]{12,}\b"), "[REDACTED_SECRET]"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b"), "[REDACTED_SECRET]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}"), "Bearer [REDACTED_SECRET]"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\d[ \-]?){12,18}\d\b"), _card),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    # Indian Aadhaar (12 digits, usually grouped 4-4-4) and PAN.
    (re.compile(r"\b[2-9]\d{3}[ \-]?\d{4}[ \-]?\d{4}\b"), "[REDACTED_AADHAAR]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[REDACTED_PAN]"),
    (re.compile(r"(?<![\w+])\+?\(?\d[\d \-().]{7,}\d\b"), _phone),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[REDACTED_IP]"),
]


def redact(text: str) -> str:
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_obj(value: Any) -> Any:
    """Recursively redact every string inside dicts/lists."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_obj(v) for v in value]
    return value

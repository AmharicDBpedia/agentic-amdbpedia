"""Deterministic input guardrails shared by every assistant entry point."""

from __future__ import annotations

import re
from typing import Protocol


class InjectionClassifier(Protocol):
    def classify(self, text: str) -> str: ...


INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore\s+(all|any|previous|prior)\s+instructions",
        r"disregard\s+(all|previous|prior)\s+instructions",
        r"system\s+prompt",
        r"developer\s+message",
        r"you\s+are\s+now",
        r"act\s+as\s+(a\s+)?system",
        r"bypass\s+(the\s+)?tools?",
        r"do\s+not\s+use\s+find_semantic_match",
        r"write\s+raw\s+xml",
        r"override\s+(the\s+)?rules?",
        r"prioritize\s+my\s+instructions",
        r"ignore\s+tool\s+results",
        r"inisitirakishini\s+ignore",
    )
]


def is_injection_attempt(text: str, classifier: InjectionClassifier | None = None) -> bool:
    """Reject known override attempts before retrieval or model calls."""

    if any(pattern.search(text) for pattern in INJECTION_PATTERNS):
        return True
    if not re.search(r"[a-z]", text, re.IGNORECASE):
        return False
    if classifier is None:
        return False
    return classifier.classify(text) == "injection"

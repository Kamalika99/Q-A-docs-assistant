"""Input validation and security guardrails."""

import re
import unicodedata
from typing import Optional, Tuple
from config import MAX_QUESTION_LENGTH

_SECRET_NOUN = (
    r"(?:api[\s-]?keys?|secret(?:\s+keys?)?s?|(?:access\s+|auth\w*\s+|bearer\s+)?tokens?|"
    r"private\s+keys?|credentials?|passcodes?|ssn|social\s+security(?:\s+numbers?)?|"
    r"credit\s+card(?:\s+numbers?)?)"
)
_REQUEST_VERB = (
    r"(?:give|show|tell|share|reveal|send|list|display|print|dump|leak|expose|provide|"
    r"disclose|what(?:'s|\s+is|\s+are)|whats)"
)
RESTRICTED_PATTERNS = [
    # asking for a secret value: "show me the API key", "what is the admin token"
    re.compile(rf"\b{_REQUEST_VERB}\b.{{0,60}}\b{_SECRET_NOUN}\b", re.I),
    # private information about people
    re.compile(r"\b(?:private|confidential|sensitive|personal)\s+(?:information|info|data|details|records)\b", re.I),
    re.compile(r"\b(?:other|another|all|every|someone|somebody)\s+(?:users?|customers?|admins?|members?|"
               r"people|accounts?)'?s?\s+(?:data|emails?|information|details|records|passwords?|phone|address)", re.I),
    # attempts to override the assistant
    re.compile(r"\b(?:ignore|disregard|forget|override)\b.{0,30}\b(?:previous|prior|above|earlier|all|your)\b"
               r".{0,20}\b(?:instructions?|rules?|prompts?)\b", re.I),
    re.compile(r"\bsystem\s+prompt\b|\bjailbreak\b|\bdeveloper\s+mode\b|\byou\s+are\s+now\b", re.I),
]

PASSWORD_REQUEST_PATTERNS = [
    re.compile(rf"\b{_REQUEST_VERB}\b.{{0,60}}\b(?:admin(?:istrator)?|root|default|master|shared|everyone|"
               r"someone|somebody|users?|customers?)'?s?\s+passwords?\b", re.I),
    re.compile(rf"\b{_REQUEST_VERB}\b.{{0,60}}\b(?!what\b|that\b|it\b|here\b|there\b|how\b|who\b)\w+'s\s+passwords?\b", re.I),
    re.compile(rf"\b{_REQUEST_VERB}\b(?:\s+me)?\s+(?:the|an?|all|their|his|her)\s+(?:\w+\s+){{0,2}}passwords?\b", re.I),
]
_RESET_CONTEXT = re.compile(r"\b(?:reset|resetting|forgot|forgotten|change|changing|recover|recovery|expire[sd]?|policy|requirements?)\b", re.I)


def validate_question(question: str) -> Tuple[str, Optional[str]]:
    """Return (cleaned_question, problem). problem is None when the input is acceptable."""
    if not isinstance(question, str):
        return "", "Please enter your question as text."
    text = unicodedata.normalize("NFKC", question)
    text = "".join(ch if (ch.isprintable() or ch in "\n\t") else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "", "Please enter a question."
    if len(text) > MAX_QUESTION_LENGTH:
        return "", (f"Your question is too long ({len(text)} characters). "
                    f"Please keep it under {MAX_QUESTION_LENGTH} characters.")
    if not re.search(r"[A-Za-z0-9]", text):
        return "", "Please enter a question using letters or numbers."
    return text, None


def is_restricted_query(text: str) -> bool:
    """True for obvious requests for credentials, secrets, private data or prompt override."""
    if any(p.search(text) for p in RESTRICTED_PATTERNS):
        return True
    if _RESET_CONTEXT.search(text):
        return False
    return any(p.search(text) for p in PASSWORD_REQUEST_PATTERNS)

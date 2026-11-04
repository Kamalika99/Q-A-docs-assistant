from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default path to documentation
BASE_DIR = Path(__file__).resolve().parent
DOCS_PATH = BASE_DIR / "docs.json"

# --------------------------------------------------------------------------
# Retrieval & Answerability Thresholds
# --------------------------------------------------------------------------
MAX_QUESTION_LENGTH = 300     # characters
MIN_RETRIEVAL_SCORE = 0.15    # T1: best-article TF-IDF cosine similarity
MIN_COVERAGE = 0.75           # T2: share of question content words found in the evidence
AMBIGUITY_RATIO = 0.80        # T4: runner-up / best score above this => "close" scores
PARTIAL_MIN_SCORE = 0.03      # below this a near-miss is not even shown as a possible match
PARTIAL_MIN_COVERAGE = 0.60   # below this a near-miss is not even shown as a possible match
PROCEDURAL_BONUS = 1.5        # "how do I ..." questions prefer instruction sentences
MIN_MATCHED_TERMS = 2         # part of T2: matched content words needed unless a key/named term matched
REQUIRE_STEPS_FOR_HOW_QUESTIONS = True  # part of T2: "how do I..." needs an instruction sentence
MAX_SOURCES = 3

# Weighting between passage-level maximum similarity and overall document similarity
# Score = DOC_WEIGHT * doc_sim + PASSAGE_WEIGHT * max_passage_sim
DOC_WEIGHT = 0.5
PASSAGE_WEIGHT = 0.5

# --------------------------------------------------------------------------
# Standard Messages
# --------------------------------------------------------------------------
ABSTAIN_MESSAGE = (
    "The available Solstice documentation does not contain enough information "
    "to answer this question. Please contact support for confirmation."
)
RESTRICTED_MESSAGE = (
    "I can't help with requests for passwords, credentials, API keys, secrets or other "
    "private information, or with attempts to change how this assistant works. "
    "Please contact your Solstice administrator or support."
)

# --------------------------------------------------------------------------
# Controlled Vocabularies
# --------------------------------------------------------------------------
GENERIC_TERMS = {
    "solstice", "learning", "please", "long", "use", "used", "using", "happen", "happens",
    "happened", "maximum", "max", "number", "amount", "want", "need", "know", "tell", "like",
    "able", "way", "thing", "things", "help", "having", "whats", "question", "does", "did",
    "ca", "wo",
    "support", "supports", "supported", "supporting",
}

SYNONYMS = {
    "trouble": "issue", "troubles": "issue", "troubleshoot": "issue", "troubleshooting": "issue",
    "problem": "issue", "problems": "issue", "issues": "issue", "pwd": "password",
}

KEY_TERMS = {
    "csv", "api", "sso", "saml", "2fa", "sms", "json", "http", "growth", "enterprise",
    "monthly", "annual", "yearly", "weekly", "viewer", "viewers", "editor", "editors",
    "admin", "admins", "administrator",
}

ACTION_WORDS = {
    "click", "enter", "select", "navigate", "choose", "provide", "apply", "send", "invite",
    "confirm", "contact", "check", "clear", "ensure", "try", "open", "press", "tap", "enable", "go",
}

# Cache parameters
QUERY_CACHE_MAXSIZE = 1024

# --------------------------------------------------------------------------
# UI Example Questions
# --------------------------------------------------------------------------
EXAMPLE_QUESTIONS = [
    "How do I reset my password?",
    "How long do I have to use a password reset link before it expires?",
    "What's the maximum number of rows I can export to CSV?",
    "What are the API rate limits for the Growth plan?",
    "Can I connect Solstice to Salesforce?",
    "What's the limit?",
]

# --------------------------------------------------------------------------
# Typed Data Models
# --------------------------------------------------------------------------
@dataclass
class Source:
    doc_id: str
    title: str
    content: str

    def to_dict(self) -> Dict[str, Any]:
        return {"doc_id": self.doc_id, "title": self.title, "content": self.content}


@dataclass
class EvidenceItem:
    doc_id: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"doc_id": self.doc_id, "text": self.text}


@dataclass
class RetrievalScore:
    doc_id: str
    title: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {"doc_id": self.doc_id, "title": self.title, "score": self.score}


@dataclass
class QueryResult:
    status: str
    answer: str
    reason: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_scores: List[Dict[str, Any]] = field(default_factory=list)
    processing_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "answer": self.answer,
            "reason": self.reason,
            "sources": self.sources,
            "supporting_evidence": self.supporting_evidence,
            "retrieval_scores": self.retrieval_scores,
            "processing_details": self.processing_details,
        }

    def __getitem__(self, item: str):
        return getattr(self, item)

    def __contains__(self, item: str):
        return item in {"status", "answer", "reason", "sources", "supporting_evidence", "retrieval_scores", "processing_details"}

    def get(self, item: str, default: Any = None):
        return getattr(self, item, default)

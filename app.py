from config import (
    ABSTAIN_MESSAGE,
    AMBIGUITY_RATIO,
    DOCS_PATH,
    MAX_QUESTION_LENGTH,
    MAX_SOURCES,
    MIN_COVERAGE,
    MIN_MATCHED_TERMS,
    MIN_RETRIEVAL_SCORE,
    PARTIAL_MIN_COVERAGE,
    PARTIAL_MIN_SCORE,
    PROCEDURAL_BONUS,
    REQUIRE_STEPS_FOR_HOW_QUESTIONS,
    RESTRICTED_MESSAGE,
    QueryResult,
)
from engine import answer_question, rank_documents, select_evidence
from index import DocsLoadError, build_index, get_index, load_documents
from nlp import (
    analyze_text,
    content_terms,
    important_terms,
    is_procedural,
    normalize_text,
    split_sentences,
    tokenize,
)
from security import is_restricted_query, validate_question

def main():
    from ui import run_app
    run_app()


if __name__ == "__main__":
    main()

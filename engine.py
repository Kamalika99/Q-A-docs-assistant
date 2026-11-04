"""Retrieval engine: ranks documents via dual document + passage aggregation and evidence selection."""

from typing import Any, Dict, List, Set, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    ABSTAIN_MESSAGE,
    AMBIGUITY_RATIO,
    DOC_WEIGHT,
    MAX_SOURCES,
    MIN_COVERAGE,
    MIN_MATCHED_TERMS,
    MIN_RETRIEVAL_SCORE,
    PARTIAL_MIN_COVERAGE,
    PARTIAL_MIN_SCORE,
    PASSAGE_WEIGHT,
    PROCEDURAL_BONUS,
    REQUIRE_STEPS_FOR_HOW_QUESTIONS,
    RESTRICTED_MESSAGE,
    QueryResult,
    Source,
)
from index import get_index
from nlp import content_terms, important_terms, is_procedural
from security import is_restricted_query, validate_question


def rank_documents(question: str, index: Dict[str, Any]) -> List[Tuple[int, float]]:
    """Stage 1: rank articles using hybrid document + passage/sentence-level aggregated scoring.

    For each document:
      - doc_score: whole-document cosine similarity
      - passage_score: maximum cosine similarity across any individual sentence in the document
      - final_score: DOC_WEIGHT * doc_score + PASSAGE_WEIGHT * passage_score

    This ensures specific buried facts get high prominence without getting diluted
    by lengthy documents, while preserving global topic context.
    """
    query_vector = index["vectorizer"].transform([question])
    doc_sims = cosine_similarity(query_vector, index["doc_matrix"]).ravel()

    scores = []
    for doc_idx, s_matrix in enumerate(index["sentence_matrices"]):
        doc_score = float(doc_sims[doc_idx])
        if s_matrix.shape[0] > 0:
            passage_sims = cosine_similarity(query_vector, s_matrix).ravel()
            best_passage_score = float(np.max(passage_sims))
        else:
            best_passage_score = doc_score

        # Aggregated score combining passage and document levels
        final_score = (DOC_WEIGHT * doc_score) + (PASSAGE_WEIGHT * best_passage_score)
        scores.append((doc_idx, final_score))

    return sorted(scores, key=lambda pair: (-pair[1], pair[0]))


def select_evidence(
    index: Dict[str, Any],
    doc_index: int,
    stems: List[str],
    procedural: bool,
    question: str,
) -> Dict[str, Any]:
    """Stage 2: Select the best one or two sentences of an article.

    The best sentence is chosen by query-term overlap (plus a small TF-IDF tie-break and,
    for 'how do I' questions, a bonus for instruction sentences). A second, adjacent
    sentence is added only when it covers question terms the first one missed.
    """
    sentences = index["sentences"][doc_index]
    wanted = set(stems)
    query_vector = index["vectorizer"].transform([question])
    sims = cosine_similarity(query_vector, index["sentence_matrices"][doc_index]).ravel()

    def score(j: int) -> float:
        overlap = len(wanted & sentences[j]["stems"])
        bonus = PROCEDURAL_BONUS if (procedural and overlap and sentences[j]["has_action"]) else 0.0
        return overlap + 0.5 * float(sims[j]) + bonus

    best = max(range(len(sentences)), key=lambda j: (score(j), -j))
    chosen = [best]
    covered = wanted & sentences[best]["stems"]
    if covered != wanted:
        neighbours = [j for j in (best - 1, best + 1) if 0 <= j < len(sentences)]
        gains = {j: len((wanted - covered) & sentences[j]["stems"]) for j in neighbours}
        if gains and max(gains.values()) > 0:
            chosen.append(max(neighbours, key=lambda j: (gains[j], -j)))
    chosen.sort()
    stem_union = set().union(*(sentences[j]["stems"] for j in chosen))
    return {
        "sentences": [sentences[j]["text"] for j in chosen],
        "positions": chosen,
        "stems": stem_union,
    }


def _make_result(
    status: str,
    answer: str,
    reason: str,
    sources: List[Dict[str, Any]] = None,
    evidence: List[Dict[str, Any]] = None,
    scores: List[Dict[str, Any]] = None,
    details: Dict[str, Any] = None,
) -> Dict[str, Any]:
    return QueryResult(
        status=status,
        answer=answer,
        reason=reason,
        sources=sources or [],
        supporting_evidence=evidence or [],
        retrieval_scores=scores or [],
        processing_details=details or {},
    ).to_dict()


def _source_dict(doc: Dict[str, str]) -> Dict[str, str]:
    return {"doc_id": doc["doc_id"], "title": doc["title"], "content": doc["content"]}


def answer_question(question: str, index: Dict[str, Any] = None) -> Dict[str, Any]:
    """Single entry point to answer a documentation question, grounded strictly in facts.

    status is one of:
      - ANSWER: verified answer with exact supporting evidence
      - PARTIAL: partial match / missing details
      - AMBIGUOUS: matches multiple articles equally well
      - ABSTAIN: documentation does not contain enough information
      - RESTRICTED: credentials or prompt override attempts
      - INVALID: malformed or empty input
    """
    # 1. Validation and security checks
    cleaned, problem = validate_question(question)
    if problem:
        return _make_result("INVALID", problem, problem)

    if is_restricted_query(cleaned):
        return _make_result(
            "RESTRICTED",
            RESTRICTED_MESSAGE,
            "The request asks for credentials, secrets or private information, "
            "or tries to change how the assistant works.",
        )

    if index is None:
        index = get_index()

    # 2. Analyze question terms
    terms = content_terms(cleaned)
    stems = [stem for _, stem in terms]
    word_of = {stem: word for word, stem in terms}
    details: Dict[str, Any] = {
        "question": cleaned,
        "content_words": [w for w, _ in terms],
        "stems": stems,
    }

    if not stems:
        details["note"] = "No meaningful content words after removing stopwords and generic terms."
        return _make_result("ABSTAIN", ABSTAIN_MESSAGE, details["note"], details=details)

    # 3. Stage 1: Document retrieval with dual passage & doc score aggregation
    ranking = rank_documents(cleaned, index)
    docs = index["docs"]
    score_of = dict(ranking)
    retrieval_scores = [
        {"doc_id": docs[i]["doc_id"], "title": docs[i]["title"], "score": round(float(s), 4)}
        for i, s in ranking
    ]
    details["ranked_doc_ids"] = [docs[i]["doc_id"] for i, _ in ranking]

    best_i, best_score = ranking[0]
    if best_score <= 0:
        reason = "None of the question's content words appear in any article."
        return _make_result("ABSTAIN", ABSTAIN_MESSAGE, reason, scores=retrieval_scores, details=details)

    # 4. Stage 2: Sentence-level evidence from the top article
    procedural = is_procedural(cleaned)
    evidence = select_evidence(index, best_i, stems, procedural, cleaned)
    details["procedural_question"] = procedural
    details["evidence_sentence_numbers"] = [n + 1 for n in evidence["positions"]]

    # 5. Answerability evaluation gates
    covered = evidence["stems"] & set(stems)
    coverage = len(covered) / len(stems)
    missing = [word_of[s] for s in stems if s not in covered]

    important = important_terms(cleaned, terms)
    unmatched_important = sorted(important - evidence["stems"])
    absent_named = [s for s in unmatched_important if not s.isdigit() and s not in index["vocab"]]

    candidates = [i for i, _ in ranking if set(stems) <= index["doc_stems"][i]]
    ambiguous = len(candidates) >= 2 and (
        len(stems) == 1 or score_of[candidates[1]] >= AMBIGUITY_RATIO * score_of[candidates[0]]
    )

    enough_terms = len(covered) >= MIN_MATCHED_TERMS or bool(covered & important)
    steps_ok = (
        not REQUIRE_STEPS_FOR_HOW_QUESTIONS
        or not procedural
        or any(index["sentences"][best_i][j]["has_action"] for j in evidence["positions"])
    )

    t1 = best_score >= MIN_RETRIEVAL_SCORE
    t2 = coverage >= MIN_COVERAGE and enough_terms and steps_ok
    t3 = not unmatched_important
    t4 = not ambiguous

    details["checks"] = {
        "T1_retrieval_score": {"passed": t1, "value": round(float(best_score), 4), "threshold": MIN_RETRIEVAL_SCORE},
        "T2_evidence_coverage": {
            "passed": t2,
            "value": round(coverage, 3),
            "threshold": MIN_COVERAGE,
            "missing_words": missing,
            "enough_matched_terms": enough_terms,
            "steps_present": steps_ok,
        },
        "T3_important_terms": {
            "passed": t3,
            "important": sorted(word_of.get(s, s) for s in important),
            "unmatched": [word_of.get(s, s) for s in unmatched_important],
        },
        "T4_ambiguity": {"passed": t4, "candidate_doc_ids": [docs[i]["doc_id"] for i in candidates]},
    }

    evidence_items = [{"doc_id": docs[best_i]["doc_id"], "text": t} for t in evidence["sentences"]]

    # 6. Final outcome decision
    if t1 and t3 and coverage >= MIN_COVERAGE and not t4:
        shown = candidates[:MAX_SOURCES]
        evidence_items = []
        for i in shown:
            ev = select_evidence(index, i, stems, procedural, cleaned)
            evidence_items += [{"doc_id": docs[i]["doc_id"], "text": t} for t in ev["sentences"]]
        names = "; ".join(f"{docs[i]['title']} ({docs[i]['doc_id']})" for i in shown)
        return _make_result(
            "AMBIGUOUS",
            f"Possible match \u2014 not verified. Your question could refer to more than one "
            f"article: {names}. Which one did you mean? Adding a specific term (for example "
            f"a feature or plan name) will help.",
            "More than one article matches the question equally well, and the question does "
            "not contain enough detail to tell them apart.",
            sources=[_source_dict(docs[i]) for i in shown],
            evidence=evidence_items,
            scores=retrieval_scores,
            details=details,
        )

    if t1 and t2 and t3 and t4:
        return _make_result(
            "ANSWER",
            " ".join(evidence["sentences"]),
            "All answerability checks passed (retrieval score, evidence coverage, "
            "important terms, no ambiguity).",
            sources=[_source_dict(docs[best_i])],
            evidence=evidence_items,
            scores=retrieval_scores,
            details=details,
        )

    # Named term that documentation never mentions => abstain
    if absent_named:
        words = ", ".join(word_of.get(s, s) for s in absent_named)
        reason = f"The documentation never mentions: {words}."
        return _make_result("ABSTAIN", ABSTAIN_MESSAGE, reason, scores=retrieval_scores, details=details)

    if best_score >= PARTIAL_MIN_SCORE and coverage >= PARTIAL_MIN_COVERAGE:
        near = [
            i
            for i, s in ranking[:MAX_SOURCES]
            if i == best_i or (s >= 0.6 * best_score and s >= PARTIAL_MIN_SCORE)
        ]
        if not t3:
            gap = ", ".join(word_of.get(s, s) for s in unmatched_important)
            reason = (
                f"The closest article does not mention: {gap}. "
                f"The match is on other words only, so it is not verified."
            )
            ask = f"Could you clarify what you need to know about {gap}? Otherwise please contact support."
        elif coverage < MIN_COVERAGE:
            gap = ", ".join(missing)
            reason = (
                f"Only {coverage:.0%} of the key words in the question are covered by the "
                f"closest supporting text (needed {MIN_COVERAGE:.0%}). Not covered: {gap}."
            )
            ask = f"Could you rephrase, or say more about {gap}? Otherwise please contact support."
        elif not enough_terms:
            reason = (
                "Only one key word in the question matched the closest article, which is too "
                "little to be sure the article answers it."
            )
            ask = "Could you add more detail to the question? Otherwise please contact support."
        elif not steps_ok:
            reason = (
                "The question asks how to do something, but the closest text does not "
                "describe steps for it."
            )
            ask = "Could you rephrase, or tell us what you are trying to achieve? Otherwise please contact support."
        else:
            reason = (
                f"The retrieval score {best_score:.2f} is below the minimum "
                f"{MIN_RETRIEVAL_SCORE:.2f} needed for a verified answer."
            )
            ask = "Could you rephrase the question with more specific terms?"

        return _make_result(
            "PARTIAL",
            f"Possible match \u2014 not verified. The closest article is "
            f"\u201c{docs[best_i]['title']}\u201d ({docs[best_i]['doc_id']}), but it does not "
            f"fully answer your question. {ask}",
            reason,
            sources=[_source_dict(docs[i]) for i in near],
            evidence=evidence_items,
            scores=retrieval_scores,
            details=details,
        )

    reason = (
        "No article contains enough of the question's key words "
        f"(best coverage {coverage:.0%}, best retrieval score {best_score:.2f})."
    )
    return _make_result("ABSTAIN", ABSTAIN_MESSAGE, reason, scores=retrieval_scores, details=details)

"""Streamlit UI components and presentation layer for Solstice Docs Assistant."""

import re
from typing import Any, Dict

import streamlit as st

from config import EXAMPLE_QUESTIONS
from engine import answer_question
from index import DocsLoadError, get_index

STATUS_STYLE = {
    "ANSWER": ("\u2705 Verified answer", "#1a7f37"),
    "PARTIAL": ("\u26a0\ufe0f Possible match \u2014 not verified", "#b7791f"),
    "AMBIGUOUS": ("\u2753 Ambiguous question \u2014 possible matches, not verified", "#1f6feb"),
    "ABSTAIN": ("\U0001F6AB Not covered by the documentation", "#6e7781"),
    "RESTRICTED": ("\U0001F512 Restricted request", "#cf222e"),
    "INVALID": ("\u270f\ufe0f Please check your question", "#8250df"),
}


def escape_markdown(text: str) -> str:
    """Escape Markdown/LaTeX special characters so user/doc text displays safely."""
    return re.sub(r"([\\`*_{}\[\]()#+!|<>~$])", r"\\\1", text)


def render_result(result: Dict[str, Any], index: Dict[str, Any]) -> None:
    """Render structured query outcome in the Streamlit UI."""
    label, color = STATUS_STYLE.get(result["status"], ("Result", "#6e7781"))
    st.markdown(
        f"<div style='border-left:6px solid {color};background:{color}14;padding:0.6rem 1rem;"
        f"border-radius:4px;font-weight:600;font-size:1.05rem'>{label}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(escape_markdown(result["answer"]))
    if result["status"] in ("PARTIAL", "AMBIGUOUS", "ABSTAIN"):
        st.caption(f"Why: {result['reason']}")

    docs_by_id = {d["doc_id"]: d for d in index["docs"]}
    if result["sources"]:
        st.markdown("**Source**" if result["status"] == "ANSWER" else "**Possible matches (not verified)**")
        for src in result["sources"]:
            st.markdown(f"\U0001F4C4 **{escape_markdown(src['doc_id'])}** \u00b7 {escape_markdown(src['title'])}")
            for ev in result["supporting_evidence"]:
                if ev["doc_id"] == src["doc_id"]:
                    st.markdown("> " + escape_markdown(ev["text"]))
            with st.expander(f"View full article \u2014 {src['doc_id']}: {src['title']}"):
                st.markdown(escape_markdown(docs_by_id[src["doc_id"]]["content"]))

    if result["retrieval_scores"]:
        st.markdown("**Retrieval scores**")
        st.caption(
            "Aggregated TF-IDF score (document + sentence-level passage similarity). "
            "This is a retrieval score, not the probability that an answer is correct."
        )
        for row in result["retrieval_scores"][:3]:
            st.markdown(f"- {escape_markdown(row['doc_id'])} \u00b7 {escape_markdown(row['title'])}: `{row['score']:.3f}`")

    checks = result["processing_details"].get("checks")
    with st.expander("How this answer was found"):
        st.markdown(
            "1. Your question is cleaned, stopwords are removed, and words are stemmed.\n"
            "2. Articles are ranked via dual passage and document TF-IDF similarity.\n"
            "3. Sentences within the best article are ranked, preserving the best evidence.\n"
            "4. Four strict answerability checks (T1-T4) verify if it is safe to answer."
        )
        if checks:
            t1, t2 = checks["T1_retrieval_score"], checks["T2_evidence_coverage"]
            t3, t4 = checks["T3_important_terms"], checks["T4_ambiguity"]
            mark = lambda ok: "\u2705" if ok else "\u274c"
            st.markdown(
                f"- {mark(t1['passed'])} **T1 retrieval score** {t1['value']} (minimum {t1['threshold']})\n"
                f"- {mark(t2['passed'])} **T2 evidence coverage** {t2['value']:.0%} (minimum {t2['threshold']:.0%})"
                + (f", missing: {escape_markdown(', '.join(t2['missing_words']))}" if t2["missing_words"] else "")
                + ("" if t2["enough_matched_terms"] else " \u2014 too few matched words")
                + ("" if t2["steps_present"] else " \u2014 no steps found for a \u201chow do I\u201d question")
                + "\n"
                f"- {mark(t3['passed'])} **T3 important terms** "
                + (f"unmatched: {escape_markdown(', '.join(t3['unmatched']))}" if t3["unmatched"] else "all matched")
                + "\n"
                f"- {mark(t4['passed'])} **T4 ambiguity** "
                + (
                    "no competing article"
                    if t4["passed"]
                    else "competing articles: " + escape_markdown(", ".join(t4["candidate_doc_ids"]))
                )
            )
        else:
            st.markdown("The question was stopped before retrieval (validation or security safeguard).")


def run_app():
    """Streamlit entry point."""
    st.set_page_config(page_title="Solstice Docs Assistant", page_icon="\U0001F4D8", layout="centered")
    st.title("Solstice Docs Assistant")
    st.write("Get reliable answers from Solstice Learning documentation.")

    try:
        index = st.cache_resource(show_spinner=False)(get_index)()
    except DocsLoadError as exc:
        st.error(f"The documentation could not be loaded: {exc}")
        return

    for warning in index.get("warnings", []):
        st.warning(warning)

    st.session_state.setdefault("question", "")
    st.session_state.setdefault("asked", None)

    def ask_example(text: str):
        st.session_state["question"] = text
        st.session_state["asked"] = text

    with st.form("ask_form"):
        st.text_input("Your question", key="question", placeholder="e.g. How do I invite a team member?")
        if st.form_submit_button("Ask", type="primary"):
            st.session_state["asked"] = st.session_state["question"]

    st.caption("Try an example:")
    columns = st.columns(2)
    for n, example in enumerate(EXAMPLE_QUESTIONS):
        columns[n % 2].button(
            example,
            key=f"example_{n}",
            on_click=ask_example,
            args=(example,),
            use_container_width=True,
        )

    if st.session_state["asked"] is not None:
        st.divider()
        result = answer_question(st.session_state["asked"], index=index)
        render_result(result, index)

    st.divider()
    st.caption(
        "Answers come only from the supplied Solstice documentation. This prototype has no "
        "authentication or per-user access control; its input filter is a basic safeguard."
    )

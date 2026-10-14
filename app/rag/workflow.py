"""The Agentic RAG workflow: graph nodes, routing decisions, and the compiled LangGraph.

Nodes read the state, do one job, and return ONLY the fields they change.
Outside services (LLMs, retriever, web) arrive through `runtime.context` as a `Services`
object, so tests can pass fakes instead of real APIs.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy
from pydantic import BaseModel

from app.core.config import Settings
from app.rag.prompts import (
    ANSWER_PROMPT,
    ANSWER_SYSTEM,
    CHECK_PROMPT,
    DIRECT_PROMPT,
    GRADER_PROMPT,
    REWRITE_PROMPT,
    ROUTER_PROMPT,
    format_history,
    format_sources,
)
from app.rag.retrieval import HybridRetriever, WebSearcher
from app.rag.state import AgentState, AnswerCheck, DocumentGrades, RouteDecision, SearchQueries

logger = logging.getLogger(__name__)


@dataclass
class Services:
    """Everything the graph needs from the outside world."""

    settings: Settings
    llm_fast: Any  # small structured decisions
    llm_strong: Any  # final answers
    retriever: HybridRetriever
    web: Optional[WebSearcher]  # None → web search disabled


# ============================================================================
# Helpers: safe structured calls + private-data check
# ============================================================================


def structured_call(llm, schema: type[BaseModel], prompt: str, default):
    """Call `llm` with structured output; return `default` instead of raising on failure.

    With OpenAI this sends a strict JSON schema, and the reply is validated by Pydantic.
    """
    try:
        result = llm.with_structured_output(schema).invoke(prompt)
        return result if result is not None else default
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash the graph
        logger.warning("%s call failed, using default: %s", schema.__name__, exc)
        return default


# Defence in depth: the LLM router also flags private data, but this regex does not
# depend on the model noticing. If either check fires, the question never goes to the web.
PRIVATE_DATA_RE = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+"  # email address
    r"|\b(?:account|acct|customer|employee|patient|invoice|order|ssn|id)\s*(?:#|no\.?|number)?\s*:?\s*\d{3,}"
    r"|(?<!\w)\+?(?:\d[\s().-]?){9,}\d(?!\w)"  # phone-number-like (10+ digits)
    r"|\b(?:password|api[_ ]?key|secret|token)\b\s*[:=]",
    re.IGNORECASE,
)


def looks_private(text: str) -> bool:
    return bool(PRIVATE_DATA_RE.search(text))


# ============================================================================
# Nodes
# ============================================================================


def trace(state: AgentState, message: str) -> list[str]:
    logger.info(message)
    return state.get("trace", []) + [message]


def _history(state: AgentState, services: Services) -> str:
    # messages[-1] is the current question; history is everything before it
    return format_history(state.get("messages", [])[:-1], services.settings.max_history_messages)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_question(state: AgentState, runtime: Runtime[Services]):
    s = runtime.context
    # Fail safe: if the router breaks, assume private data so nothing leaks to the web.
    fallback = RouteDecision(route="retrieve", standalone_question=state["question"], contains_private_data=True)
    decision = structured_call(
        s.llm_fast,
        RouteDecision,
        ROUTER_PROMPT.format(
            kb_description=s.settings.kb_description, history=_history(state, s), question=state["question"]
        ),
        default=fallback,
    )
    standalone = decision.standalone_question.strip() or state["question"]
    web_allowed = (
        s.web is not None
        and s.settings.enable_web
        and not decision.contains_private_data
        and not looks_private(state["question"])
    )
    return {
        "route": decision.route,
        "standalone_question": standalone,
        "queries": [standalone],
        "web_allowed": web_allowed,
        "trace": trace(state, f"route={decision.route} | web_allowed={web_allowed} | standalone='{standalone}'"),
    }


# ---------------------------------------------------------------------------
# Knowledge base path
# ---------------------------------------------------------------------------


def retrieve_kb(state: AgentState, runtime: Runtime[Services]):
    queries = state.get("queries") or [state["standalone_question"]]
    docs = runtime.context.retriever.retrieve(queries)
    attempt = state.get("kb_attempts", 0) + 1
    scores = [d.metadata.get("rerank_score") for d in docs]
    return {
        "kb_docs": docs,
        "kb_attempts": attempt,
        "trace": trace(
            state, f"KB attempt {attempt}: {len(queries)} query(ies) → {len(docs)} reranked chunks {scores}"
        ),
    }


def grade_documents(llm, question: str, docs: list[Document]) -> tuple[list[Document], bool]:
    """CRAG-style grading: returns (relevant_docs, sufficient)."""
    if not docs:
        return [], False
    grades = structured_call(
        llm,
        DocumentGrades,
        GRADER_PROMPT.format(question=question, sources=format_sources(docs, max_chars=1500)),
        default=None,
    )
    if grades is None:  # grader unavailable → fall back to the reranker's judgement
        return [d for d in docs if d.metadata.get("rerank_score", 0) >= 0.5], False
    keep = {i for i in grades.relevant_ids if 1 <= i <= len(docs)}
    relevant = [d for i, d in enumerate(docs, 1) if i in keep]
    return relevant, bool(relevant) and grades.sufficient


def grade_kb(state: AgentState, runtime: Runtime[Services]):
    kb_docs = state.get("kb_docs", [])
    relevant, sufficient = grade_documents(runtime.context.llm_fast, state["standalone_question"], kb_docs)
    if relevant and sufficient:
        decision = "correct"
    elif relevant:
        decision = "ambiguous"
    else:
        decision = "incorrect"
    return {
        "evidence": relevant,
        "crag_decision": decision,
        "trace": trace(state, f"KB grade: {len(relevant)}/{len(kb_docs)} relevant → {decision}"),
    }


def rewrite_query(state: AgentState, runtime: Runtime[Services]):
    s = runtime.context
    question = state["standalone_question"]
    tried = state.get("queries", [question])
    result = structured_call(
        s.llm_fast,
        SearchQueries,
        REWRITE_PROMPT.format(kb_description=s.settings.kb_description, question=question, tried=tried),
        default=SearchQueries(queries=[]),
    )
    new_queries: list[str] = []
    for q in (q.strip() for q in result.queries):
        if q and q not in tried and q not in new_queries:  # skip empty, already-tried, duplicates
            new_queries.append(q)
    new_queries = new_queries[:3]
    queries = [question] + new_queries  # the original stays first: it is used for reranking
    return {"queries": queries, "trace": trace(state, f"multi-query rewrite → {new_queries}")}


# ---------------------------------------------------------------------------
# Web path
# ---------------------------------------------------------------------------


def web_search(state: AgentState, runtime: Runtime[Services]):
    s = runtime.context
    query = state["standalone_question"]
    docs = s.retriever.rerank(query, s.web.search(query)) if s.web else []
    return {
        "web_docs": docs,
        "web_searched": True,
        "trace": trace(state, f"web search → {len(docs)} reranked results"),
    }


def grade_web(state: AgentState, runtime: Runtime[Services]):
    web_docs = state.get("web_docs", [])
    relevant, _ = grade_documents(runtime.context.llm_fast, state["standalone_question"], web_docs)
    evidence = state.get("evidence", []) + relevant  # keep any partial KB evidence too
    return {
        "evidence": evidence,
        "trace": trace(state, f"web grade: {len(relevant)}/{len(web_docs)} relevant"),
    }


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------

CITATION_RE = re.compile(r"(\s?)\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citations(answer: str, evidence: list[Document]) -> tuple[str, list[dict]]:
    """Keep only citation numbers that point at real sources; build the source list."""
    used: set[int] = set()

    def _clean(match: re.Match) -> str:
        valid = [n for n in (int(x) for x in match.group(2).split(",")) if 1 <= n <= len(evidence)]
        used.update(valid)
        return match.group(1) + "[" + ", ".join(map(str, valid)) + "]" if valid else ""

    answer = CITATION_RE.sub(_clean, answer)
    citations = []
    for n in sorted(used):
        meta = evidence[n - 1].metadata
        citations.append(
            {
                "id": n,
                "type": "Private KB" if meta.get("source_type") == "kb" else "Web",
                "title": meta.get("title", ""),
                "section": meta.get("section", ""),
                "url": meta.get("source", ""),
            }
        )
    return answer, citations


def generate(state: AgentState, runtime: Runtime[Services]):
    s = runtime.context
    feedback = ""
    check = state.get("check") or {}
    if check and not check.get("grounded", True):
        claims = "; ".join(check.get("unsupported_claims", [])) or "some claims"
        feedback = f"\nA fact-checker found these claims NOT supported by the sources. Remove or correct them: {claims}"
    response = s.llm_strong.invoke(
        [
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(
                content=ANSWER_PROMPT.format(
                    history=_history(state, s),
                    question=state["standalone_question"],
                    sources=format_sources(state["evidence"], s.settings.max_source_chars),
                    feedback=feedback,
                )
            ),
        ]
    )
    answer, citations = extract_citations(response.text, state["evidence"])  # .text: plain string
    attempt = state.get("generation_attempts", 0) + 1
    return {
        "answer": answer,
        "citations": citations,
        "generation_attempts": attempt,
        "trace": trace(state, f"generated answer (attempt {attempt}) citing {[c['id'] for c in citations]}"),
    }


def check_answer(state: AgentState, runtime: Runtime[Services]):
    s = runtime.context
    result = structured_call(
        s.llm_fast,
        AnswerCheck,
        CHECK_PROMPT.format(
            question=state["standalone_question"],
            sources=format_sources(state["evidence"], s.settings.max_source_chars),
            answer=state["answer"],
        ),
        default=None,
    )
    if result is None:
        check = {"grounded": True, "answers_question": True, "unsupported_claims": [], "skipped": True}
    else:
        check = result.model_dump()
    return {
        "check": check,
        "trace": trace(
            state,
            f"self-check: grounded={check['grounded']} answers_question={check['answers_question']}"
            + (" (checker unavailable)" if check.get("skipped") else ""),
        ),
    }


def direct_answer(state: AgentState, runtime: Runtime[Services]):
    s = runtime.context
    response = s.llm_fast.invoke(DIRECT_PROMPT.format(history=_history(state, s), question=state["question"]))
    return {"answer": response.text, "trace": trace(state, "answered directly (no retrieval)")}


def insufficient(state: AgentState):
    reason = (
        "neither the knowledge base nor the web search returned relevant information"
        if state.get("web_allowed")
        else "the knowledge base had no relevant information and web search is disabled for this "
        "question (it may contain private data)"
    )
    answer = (
        f"I couldn't find enough reliable evidence to answer this: {reason}. "
        "Try rephrasing, adding detail, or adding the relevant documents to the knowledge base."
    )
    return {"answer": answer, "citations": [], "trace": trace(state, "insufficient evidence")}


def finalize(state: AgentState):
    answer = state.get("answer", "")
    check = state.get("check") or {}
    citations = state.get("citations", [])

    if check and not check.get("grounded", True):
        answer = "⚠️ Some statements below could not be fully verified against the sources.\n\n" + answer

    types = {c["type"] for c in citations}
    if state.get("route") == "direct":
        source_used = "direct"
    elif not citations:
        source_used = "insufficient_evidence" if not state.get("evidence") else "uncited"
    elif types == {"Private KB"}:
        source_used = "private_kb"
    elif types == {"Web"}:
        source_used = "web_search"
    else:
        source_used = "private_kb + web_search"

    return {
        "answer": answer,
        "source_used": source_used,
        "messages": [AIMessage(content=answer)],  # saved to chat history by the checkpointer
        "trace": trace(state, f"final source: {source_used}"),
    }


# ============================================================================
# Routing: which node runs next (every loop has a hard limit from Config)
# ============================================================================


def after_route(state: AgentState) -> Literal["direct_answer", "web_search", "retrieve_kb"]:
    if state["route"] == "direct":
        return "direct_answer"
    if state["route"] == "web" and state["web_allowed"]:
        return "web_search"
    return "retrieve_kb"  # private-first by default


def after_grade_kb(
    state: AgentState, runtime: Runtime[Services]
) -> Literal["generate", "rewrite_query", "web_search", "insufficient"]:
    decision = state["crag_decision"]
    if decision == "correct":
        return "generate"
    if decision == "incorrect" and state["kb_attempts"] < runtime.context.settings.max_kb_attempts:
        return "rewrite_query"
    if state["web_allowed"]:
        return "web_search"  # ambiguous → supplement; incorrect after rewrite → replace
    return "generate" if state.get("evidence") else "insufficient"


def after_grade_web(state: AgentState) -> Literal["generate", "insufficient"]:
    return "generate" if state.get("evidence") else "insufficient"


def after_check(state: AgentState, runtime: Runtime[Services]) -> Literal["generate", "web_search", "finalize"]:
    check = state["check"]
    if not check["grounded"] and state["generation_attempts"] < runtime.context.settings.max_generation_attempts:
        return "generate"  # retry with the checker's feedback
    if check["grounded"] and not check["answers_question"] and state["web_allowed"] and not state.get("web_searched"):
        return "web_search"  # grounded but incomplete → get more evidence once
    return "finalize"


# ============================================================================
# Build the graph
# ============================================================================


def build_graph(checkpointer=None):
    """Compile the agent graph.

    * context_schema=Services: nodes get LLMs/retriever/web via `runtime.context`.
    * RetryPolicy: automatic retries on transient network errors.
    * checkpointer: stores state per thread_id → conversation memory.
      Swap InMemorySaver for a Postgres/Redis saver in production.
    """
    network_retry = RetryPolicy(max_attempts=3, initial_interval=1.0)

    g = StateGraph(AgentState, context_schema=Services)
    g.add_node("route_question", route_question)
    g.add_node("retrieve_kb", retrieve_kb, retry_policy=network_retry)
    g.add_node("grade_kb", grade_kb)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("web_search", web_search, retry_policy=network_retry)
    g.add_node("grade_web", grade_web)
    g.add_node("generate", generate, retry_policy=network_retry)
    g.add_node("check_answer", check_answer)
    g.add_node("direct_answer", direct_answer, retry_policy=network_retry)
    g.add_node("insufficient", insufficient)
    g.add_node("finalize", finalize)

    g.add_edge(START, "route_question")
    g.add_conditional_edges("route_question", after_route)
    g.add_edge("retrieve_kb", "grade_kb")
    g.add_conditional_edges("grade_kb", after_grade_kb)
    g.add_edge("rewrite_query", "retrieve_kb")
    g.add_edge("web_search", "grade_web")
    g.add_conditional_edges("grade_web", after_grade_web)
    g.add_edge("generate", "check_answer")
    g.add_conditional_edges("check_answer", after_check)
    g.add_edge("direct_answer", "finalize")
    g.add_edge("insufficient", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer or InMemorySaver())

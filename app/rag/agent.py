"""The agent's public interface: build the LLMs and conversation store, and answer questions."""

import logging
import sqlite3
import time
from pathlib import Path

from langchain_core.messages import HumanMessage

from app.core.config import Settings
from app.rag.state import AgentState
from app.rag.workflow import Services, build_graph

logger = logging.getLogger(__name__)


def make_llms(settings: Settings):
    """Return (llm_fast, llm_strong).

    Uses OpenAI's Responses API: on current models, structured outputs and tool calling work
    there with reasoning switched on. No temperature is set (reasoning models only accept the
    default). The strong model falls back to the fast model if it fails.
    """
    from langchain_openai import ChatOpenAI

    def chat(model: str, effort: str, retries: int) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key,
            use_responses_api=True,
            reasoning={"effort": effort},
            max_retries=retries,
            timeout=settings.llm_timeout_s,
        )

    llm_fast = chat(settings.fast_model, settings.fast_reasoning_effort, retries=3)
    llm_strong = chat(settings.strong_model, settings.strong_reasoning_effort, retries=2).with_fallbacks(
        [chat(settings.fast_model, settings.fast_reasoning_effort, retries=3)]
    )
    return llm_fast, llm_strong


def make_checkpointer(settings: Settings):
    """Where conversation memory lives.

    "sqlite": a local file, so conversations survive a restart (one server process).
    "memory": RAM only, lost on restart (tests, quick demos).
    For several server replicas, swap in langgraph's Postgres checkpointer.
    """
    if settings.checkpointer == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(settings.checkpoint_db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def new_turn(question: str) -> AgentState:
    """Fresh per-turn state. `messages` is appended to the thread's history, not replaced."""
    return {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "standalone_question": question,
        "route": "",
        "web_allowed": False,
        "queries": [],
        "kb_attempts": 0,
        "kb_docs": [],
        "web_docs": [],
        "web_searched": False,
        "evidence": [],
        "crag_decision": "",
        "answer": "",
        "citations": [],
        "generation_attempts": 0,
        "check": {},
        "source_used": "",
        "trace": [],
    }


class RAGAgent:
    def __init__(self, services: Services, checkpointer=None):
        self.services = services
        self.app = build_graph(checkpointer)

    def stream(self, question: str, thread_id: str = "default"):
        """Run one turn, yielding ("step", node, trace_line) as each step finishes, then ("final", result).

        The same thread_id continues the same conversation.
        """
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.services.settings.recursion_limit,
        }
        start = time.perf_counter()
        for update in self.app.stream(new_turn(question), config, context=self.services, stream_mode="updates"):
            for node, changes in update.items():
                if changes and changes.get("trace"):
                    yield ("step", node, changes["trace"][-1])

        result = dict(self.app.get_state(config).values)
        result["latency_s"] = round(time.perf_counter() - start, 2)
        logger.info(
            "answered thread=%s route=%s source=%s latency=%.2fs",
            thread_id,
            result.get("route"),
            result.get("source_used"),
            result["latency_s"],
        )
        yield ("final", None, result)

    def ask(self, question: str, thread_id: str = "default", on_step=None) -> dict:
        """Run one turn and return the final state. on_step(node, line) is called per step."""
        result: dict = {}
        for kind, node, payload in self.stream(question, thread_id):
            if kind == "step" and on_step:
                on_step(node, payload)
            elif kind == "final":
                result = payload
        return result

    def draw(self) -> str:
        """Mermaid text of the graph (paste into mermaid.live)."""
        return self.app.get_graph().draw_mermaid()

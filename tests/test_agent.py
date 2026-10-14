"""End-to-end graph tests with fake services: every route and every safety limit."""

from langchain_core.messages import AIMessage

from app.rag.workflow import looks_private
from app.services.evaluation import run_eval

KB_QUESTION = "In Agentic RAG, what happens when retrieved documents are not relevant to the question?"


def test_kb_answer_with_clean_citations(agent):
    r = agent.ask(KB_QUESTION)
    assert r["source_used"] == "private_kb"
    assert "[7]" not in r["answer"] and "cite." in r["answer"]  # invalid citation removed cleanly
    assert {c["type"] for c in r["citations"]} == {"Private KB"}
    assert not r["web_searched"]


def test_out_of_scope_goes_to_web(agent):
    r = agent.ask("What is Tavily Search?")
    assert r["source_used"] == "web_search"
    assert all("IGNORED" not in d.page_content for d in r["evidence"])  # Tavily's AI answer is not evidence


def test_small_talk_skips_retrieval(agent, fake_llm):
    r = agent.ask("Hello, how are you?")
    assert r["source_used"] == "direct"
    assert fake_llm.calls == ["RouteDecision", "chat"]


def test_partial_kb_evidence_is_supplemented_by_web(agent, fake_llm, fake_tavily):
    fake_llm.sufficient = False
    r = agent.ask(KB_QUESTION)
    assert r["web_searched"] and fake_tavily.calls == 1
    assert r["crag_decision"] == "ambiguous"


def test_no_match_triggers_one_multi_query_rewrite(agent):
    r = agent.ask("zzz qqq unknown thing")
    assert r["kb_attempts"] == 2
    assert len(r["queries"]) == 2  # original + 1 new (duplicates removed)
    assert r["source_used"] == "insufficient_evidence"


def test_private_question_never_reaches_web(agent, fake_tavily):
    r = agent.ask("Customer account 1234: zzz qqq unknown")
    assert r["web_allowed"] is False and not r["web_searched"] and fake_tavily.calls == 0
    assert r["source_used"] == "insufficient_evidence"


def test_ungrounded_answer_is_regenerated_once(agent, fake_llm):
    fake_llm.grounded = [False, True]
    r = agent.ask(KB_QUESTION)
    assert r["generation_attempts"] == 2
    assert not r["answer"].startswith("⚠️")


def test_still_ungrounded_answer_gets_warning(agent, fake_llm):
    fake_llm.grounded = [False, False]
    r = agent.ask(KB_QUESTION)
    assert r["generation_attempts"] == 2
    assert r["answer"].startswith("⚠️")


def test_incomplete_answer_with_empty_web_does_not_loop(agent, fake_llm, fake_tavily):
    fake_tavily.empty = True
    fake_llm.answers = [False, False, False]
    r = agent.ask(KB_QUESTION)
    assert fake_tavily.calls == 1
    assert r["source_used"] == "private_kb"


def test_grader_failure_falls_back_to_reranker(agent, fake_llm):
    fake_llm.grader_fail = True
    r = agent.ask("grade documents relevant question retrieved")
    assert r["evidence"]  # reranker scores were used instead
    assert r["answer"]


def test_router_failure_fails_safe(agent, fake_llm, fake_tavily):
    fake_llm.broken = True
    r = agent.ask("What is Tavily Search?")
    assert r["web_allowed"] is False and fake_tavily.calls == 0


def test_follow_up_uses_conversation_memory(agent):
    agent.ask("How does the tutorial grade retrieved documents?", thread_id="t")
    r = agent.ask("And what does it do after that step if the grade is bad?", thread_id="t")
    assert len(r["messages"]) == 4
    assert isinstance(r["messages"][-1], AIMessage)
    assert r["standalone_question"].startswith("What happens after grading")


def test_threads_are_isolated(agent):
    agent.ask("How does the tutorial grade retrieved documents?", thread_id="a")
    r = agent.ask("Hello, how are you?", thread_id="b")
    assert len(r["messages"]) == 2


def test_web_disabled_when_no_web_service(agent, services):
    services.web = None
    r = agent.ask("What is Tavily Search?")
    assert r["web_allowed"] is False and not r["web_searched"]


def test_eval_runner(agent, capsys):
    rows = run_eval(agent)
    assert len(rows) == 5
    assert sum(r["web_leak"] for r in rows) == 0
    assert "source accuracy" in capsys.readouterr().out


def test_graph_draws(agent):
    assert "route_question" in agent.draw()


def test_responses_api_content_blocks_are_handled(agent, fake_llm):
    """OpenAI's Responses API returns content as a list of blocks, not a plain string."""
    blocks = [{"type": "text", "text": "Block answer [1].", "annotations": [], "id": "msg_1"}]
    fake_llm.invoke = lambda messages: AIMessage(content=blocks)
    r = agent.ask(KB_QUESTION, thread_id="blocks")
    assert r["answer"] == "Block answer [1]."
    assert [c["id"] for c in r["citations"]] == [1]
    r = agent.ask("Hello, how are you?", thread_id="blocks")
    assert r["answer"] == "Block answer [1]."


def test_privacy_regex():
    assert looks_private("Customer account 88231 complained")
    assert looks_private("Employee John Miller (ID 5521)")
    assert looks_private("email me at a@b.com")
    assert looks_private("call 205-555-1234")
    assert not looks_private("What happened on 2026-10-13?")
    assert not looks_private("langgraph 1.2.15 release notes")


def test_stream_yields_steps_then_final(agent):
    events = list(agent.stream(KB_QUESTION, thread_id="s"))
    kinds = [e[0] for e in events]
    assert kinds[-1] == "final" and kinds.count("final") == 1
    assert [e[1] for e in events if e[0] == "step"][0] == "route_question"
    assert events[-1][2]["source_used"] == "private_kb"


def test_sqlite_checkpointer_keeps_memory_across_restarts(services, settings):
    from app.rag.agent import RAGAgent, make_checkpointer

    settings.checkpointer = "sqlite"
    first = RAGAgent(services, checkpointer=make_checkpointer(settings))
    first.ask("How does the tutorial grade retrieved documents?", thread_id="persist")
    restarted = RAGAgent(services, checkpointer=make_checkpointer(settings))  # new process, same DB file
    r = restarted.ask("And what does it do after that step if the grade is bad?", thread_id="persist")
    assert len(r["messages"]) == 4
    assert r["source_used"] == "private_kb"

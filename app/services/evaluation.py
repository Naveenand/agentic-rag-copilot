"""A small evaluation harness: fixed questions, scored the same way after every change.

Grow EVAL_SET with real questions your users ask (and their expected sources).
"""

import time

EVAL_SET = [
    {
        "question": "In the LangGraph agentic RAG tutorial, what happens when retrieved documents are not relevant?",
        "expected_source": {"private_kb", "private_kb + web_search"},
        "keywords": ["rewrite"],
    },
    {
        "question": "How are retrieved documents graded in the LangGraph agentic RAG tutorial?",
        "expected_source": {"private_kb", "private_kb + web_search"},
        "keywords": ["relevan"],
    },
    {
        "question": "What is Tavily Search?",
        "expected_source": {"web_search"},
        "keywords": ["search"],
    },
    {
        "question": "Thanks, that was helpful!",
        "expected_source": {"direct"},
        "keywords": [],
    },
    {
        "question": "Employee John Miller (ID 5521) asked: how does the tutorial decide whether to retrieve?",
        "expected_source": {"private_kb", "insufficient_evidence"},
        "keywords": [],
        "private": True,
    },
]


def run_eval(agent, eval_set=EVAL_SET, report: bool = True) -> list[dict]:
    rows = []
    for i, case in enumerate(eval_set):
        r = agent.ask(case["question"], thread_id=f"eval-{i}-{time.time()}")
        answer = r["answer"].lower()
        rows.append(
            {
                "question": case["question"][:55],
                "source": r["source_used"],
                "source_ok": r["source_used"] in case["expected_source"],
                "keywords_ok": all(k in answer for k in case["keywords"]),
                "grounded": (r.get("check") or {}).get("grounded", True),
                "web_leak": bool(case.get("private")) and bool(r.get("web_searched")),
                "latency_s": r["latency_s"],
            }
        )
    if report:
        print_report(rows)
    return rows


def print_report(rows: list[dict]) -> None:
    def tick(ok: bool) -> str:
        return "✓" if ok else "✗"

    print(f"{'question':57}{'source':26}{'src':5}{'kw':5}{'grnd':6}{'leak':6}{'sec':>6}")
    for r in rows:
        print(
            f"{r['question']:57}{r['source']:26}{tick(r['source_ok']):5}{tick(r['keywords_ok']):5}"
            f"{tick(r['grounded']):6}{'LEAK' if r['web_leak'] else '-':6}{r['latency_s']:>6}"
        )
    n = len(rows) or 1
    print(
        f"\nsource accuracy {sum(r['source_ok'] for r in rows)}/{len(rows)} | "
        f"keyword hits {sum(r['keywords_ok'] for r in rows)}/{len(rows)} | "
        f"grounded {sum(r['grounded'] for r in rows)}/{len(rows)} | "
        f"web leaks {sum(r['web_leak'] for r in rows)} | "
        f"avg latency {sum(r['latency_s'] for r in rows) / n:.1f}s"
    )

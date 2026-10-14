"""Command-line tools.

    python cli.py ingest            # add SOURCE_URLS and every file in data/sample_kb to the knowledge base
    python cli.py ingest FILE_OR_URL [...]   # add specific files or pages
    python cli.py chat              # chat in the terminal (/new = new conversation, /quit = exit)
    python cli.py eval              # run the evaluation set
    python cli.py sources           # list documents in the knowledge base

The web server is started with `python run.py` (development) or Docker (production).
"""

import argparse
import sys
import uuid
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.services.ingestion import SUPPORTED_EXTENSIONS, IngestionError


def cmd_ingest(targets: list[str]) -> int:
    from app.bootstrap import build_knowledge_base

    settings = get_settings()
    missing = settings.missing_keys(("pinecone_api_key",))
    if missing:
        print(f"Missing {', '.join(missing)} in .env")
        return 1
    kb, _ = build_knowledge_base(settings)

    if not targets:
        targets = list(settings.source_urls) + sorted(
            str(p) for p in settings.sample_kb_dir.glob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    failures = 0
    for target in targets:
        try:
            if target.startswith(("http://", "https://")):
                r = kb.ingest_url(target)
            else:
                r = kb.ingest_file(Path(target))
            print(
                f"✓ {r.title}: {r.chunks} chunks ({r.uploaded} uploaded, {r.unchanged} unchanged, {r.removed} removed)"
            )
        except (IngestionError, FileNotFoundError) as exc:
            failures += 1
            print(f"✗ {target}: {exc}")
    print(f"Knowledge base now holds {kb.store.count()} chunks from {len(kb.store.sources())} documents.")
    return 1 if failures else 0


def _build_agent():
    from app.bootstrap import build_components

    return build_components(get_settings()).agent


def cmd_chat() -> int:
    print("Loading models and connecting to services…")
    agent = _build_agent()
    thread_id = uuid.uuid4().hex
    print("Ready. Ask a question (/new for a new conversation, /quit to exit).\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question == "/quit":
            break
        if question == "/new":
            thread_id = uuid.uuid4().hex
            print("Started a new conversation.\n")
            continue
        if not question:
            continue
        result = agent.ask(question, thread_id=thread_id, on_step=lambda node, line: print(f"  [{node}] {line}"))
        print(f"\nAssistant ({result['source_used']}, {result['latency_s']}s):\n{result['answer']}")
        for c in result.get("citations", []):
            section = f" › {c['section']}" if c["section"] else ""
            print(f"  [{c['id']}] ({c['type']}) {c['title']}{section}  {c['url']}")
        print()
    return 0


def cmd_eval() -> int:
    from app.services.evaluation import run_eval

    rows = run_eval(_build_agent())
    return 0 if all(r["source_ok"] and not r["web_leak"] for r in rows) else 1


def cmd_sources() -> int:
    from app.services.knowledge_base import ChunkStore

    store = ChunkStore(get_settings().chunk_db_path)
    for s in store.sources():
        print(f"{s['chunks']:>5} chunks  {s['title']}  ({s['source']})")
    print(f"Total: {store.count()} chunks")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Agentic RAG Knowledge Copilot tools")
    sub = parser.add_subparsers(dest="command", required=True)
    p_ingest = sub.add_parser("ingest", help="add documents to the knowledge base")
    p_ingest.add_argument("targets", nargs="*", help="files or URLs (default: SOURCE_URLS + data/sample_kb)")
    sub.add_parser("chat", help="chat in the terminal")
    sub.add_parser("eval", help="run the evaluation set")
    sub.add_parser("sources", help="list documents in the knowledge base")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging("WARNING" if args.command in {"chat", "eval"} else settings.log_level)
    if args.command == "ingest":
        return cmd_ingest(args.targets)
    if args.command == "chat":
        return cmd_chat()
    if args.command == "eval":
        return cmd_eval()
    return cmd_sources()


if __name__ == "__main__":
    sys.exit(main())

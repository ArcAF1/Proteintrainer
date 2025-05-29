

"""Command line interface for ResearchMemory."""

from __future__ import annotations

import argparse
from pathlib import Path

import json

from . import ResearchMemory


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="researchmem")
    sub = p.add_subparsers(dest="cmd", required=True)

    new_p = sub.add_parser("new")
    new_p.add_argument("type")
    new_p.add_argument("title")
    new_p.add_argument("body")

    search_p = sub.add_parser("search")
    search_p.add_argument("query")
    search_p.add_argument("-k", type=int, default=5)

    show_p = sub.add_parser("show")
    show_p.add_argument("id")

    update_p = sub.add_parser("update")

    update_p.add_argument("id")
    update_p.add_argument("--status", required=True)
    update_p.add_argument("--confidence", type=float)


    return p


def main(argv: list[str] | None = None) -> None:
    parser = get_parser()
    args = parser.parse_args(argv)
    mem = ResearchMemory(embedding_fn=lambda t: [0.0])
    if args.cmd == "new":
        mem.add_entry(type=args.type, title=args.title, body=args.body)
    elif args.cmd == "search":
        hits = mem.search(args.query, k=args.k)
        for h in hits:
            print(h.score, h.entry.id, h.entry.title)
    elif args.cmd == "show":
        rows = mem.list_entries()
        for r in rows:
            if r.id == args.id:
                print(json.dumps(r.model_dump(), indent=2))
                break
    elif args.cmd == "update":
        mem.update_status(args.id, args.status, confidence=args.confidence)



if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from .db import ResearchMemory


def main() -> None:
    parser = argparse.ArgumentParser("researchmem")
    sub = parser.add_subparsers(dest="cmd", required=True)

    new_p = sub.add_parser("new", help="create a new entry")
    new_p.add_argument("type")
    new_p.add_argument("title")

    search_p = sub.add_parser("search", help="search notes")
    search_p.add_argument("query")

    show_p = sub.add_parser("show", help="show entry")
    show_p.add_argument("id")

    update_p = sub.add_parser("update", help="update status")
    update_p.add_argument("id")
    update_p.add_argument("--status", required=True)
    update_p.add_argument("--confidence", type=float)

    args = parser.parse_args()
    mem = ResearchMemory(lambda x: [0.0])  # placeholder embedding

    if args.cmd == "new":
        body = input("Body: ")
        mem.add_entry(type=args.type, title=args.title, body=body)
    elif args.cmd == "search":
        hits = mem.search(args.query)
        for h in hits:
            print(h.score, h.entry.id, h.entry.title)
    elif args.cmd == "show":
        path = Path.home() / "research_memory" / "entries" / f"{args.id}.md"
        print(path.read_text())
    elif args.cmd == "update":
        mem.update_status(args.id, args.status, args.confidence)


if __name__ == "__main__":
    main()

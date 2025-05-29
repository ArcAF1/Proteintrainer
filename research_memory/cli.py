"""Command-line interface for research memory."""
from __future__ import annotations

import argparse

from .db import ResearchMemory, MemoryConfig
from .embeddings import default_embedding_fn


def main() -> None:
    parser = argparse.ArgumentParser(prog="researchmem")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub_new = sub.add_parser("new", help="create a new entry")
    sub_new.add_argument("type")
    sub_new.add_argument("title")
    sub_new.add_argument("body")

    sub_search = sub.add_parser("search", help="search entries")
    sub_search.add_argument("query")
    sub_search.add_argument("-k", type=int, default=5)

    sub_show = sub.add_parser("show", help="show entry")
    sub_show.add_argument("id")

    sub_update = sub.add_parser("update", help="update status")
    sub_update.add_argument("id")
    sub_update.add_argument("--status", required=True)
    sub_update.add_argument("--confidence", type=float)

    args = parser.parse_args()
    mem = ResearchMemory(embedding_fn=default_embedding_fn)

    if args.cmd == "new":
        entry = mem.add_entry(type=args.type, title=args.title, body=args.body)
        print("Created", entry.id)
    elif args.cmd == "search":
        results = mem.search(args.query, k=args.k)
        for ent, score in results:
            print(f"{ent.id}\t{ent.title}\t{score:.3f}")
    elif args.cmd == "show":
        entries = mem.list_entries()
        for e in entries:
            if e.id == args.id:
                print(e.body_md)
                break
    elif args.cmd == "update":
        entry = mem.update_status(args.id, args.status, confidence=args.confidence)
        print("Updated", entry.id)


if __name__ == "__main__":
    main()

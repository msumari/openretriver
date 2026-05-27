import argparse
import logging
import sys

from src.pipeline import ingest
from src.rag import ask
from src.session import repl


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(prog="openretriver")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("path", nargs="?", default=".")
    ingest_parser.add_argument("--name", default=None)

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question", nargs="+")
    ask_parser.add_argument("--name", default=None)

    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("--name", default=None)

    if not argv:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "ingest":
        count = ingest(args.path, collection_name=args.name)
        print(f"Ingested {count} chunks from {args.path}")
    elif args.command == "ask":
        question = " ".join(args.question)
        response = ask(question, collection_name=args.name)
        print(f"\n{response.answer}")
        if response.sources:
            print(f"\nSources:")
            for source in response.sources:
                print(f"  - {source}")
    elif args.command == "chat":
        repl(collection_name=args.name)

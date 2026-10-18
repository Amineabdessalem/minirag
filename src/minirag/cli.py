"""Command-line interface: ``minirag index``, ``minirag ask`` and ``minirag eval``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from minirag.evaluation import evaluate
from minirag.pipeline import RAGPipeline
from minirag.retrievers import BM25Retriever, HybridRetriever, VectorRetriever

RETRIEVERS = {
    "bm25": lambda: BM25Retriever(),
    "vector": lambda: VectorRetriever(),
    "hybrid": lambda: HybridRetriever([BM25Retriever(), VectorRetriever()]),
}


def _pipeline(args: argparse.Namespace) -> RAGPipeline:
    if getattr(args, "index", None):
        return RAGPipeline.load(args.index, retriever=RETRIEVERS[args.retriever]())
    rag = RAGPipeline(retriever=RETRIEVERS[args.retriever](), max_words=args.max_words)
    n = rag.add_directory(args.docs, args.glob)
    if n == 0:
        sys.exit(f"no documents matching {args.glob!r} under {args.docs}")
    return rag


def cmd_ask(args: argparse.Namespace) -> None:
    answer = _pipeline(args).ask(args.question, k=args.k)
    if args.prompt:
        print(answer.prompt)
        return
    for i, hit in enumerate(answer.hits, 1):
        snippet = " ".join(hit.chunk.text.split())[:160]
        print(f"[{i}] {hit.score:.4f}  {hit.chunk.doc_id}\n    {snippet}")


def cmd_index(args: argparse.Namespace) -> None:
    rag = _pipeline(args)
    rag.save(args.out)
    print(f"saved {len(rag.chunks)} chunks to {args.out}")


def cmd_eval(args: argparse.Namespace) -> None:
    labelled = [
        (row["query"], set(row["relevant"]))
        for row in map(json.loads, Path(args.queries).read_text().splitlines())
        if row
    ]
    for name in args.compare or [args.retriever]:
        args.retriever = name
        report = evaluate(_pipeline(args).retriever, labelled, k=args.k)
        print(f"{name:<7} {report}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="minirag", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    source = common.add_mutually_exclusive_group(required=True)
    source.add_argument("--docs", help="directory of documents to index")
    source.add_argument("--index", help="directory written by `minirag index`")
    common.add_argument("--glob", default="**/*.md", help="file pattern (default: **/*.md)")
    common.add_argument("--retriever", choices=RETRIEVERS, default="hybrid")
    common.add_argument("--max-words", type=int, default=120, help="chunk size in words")
    common.add_argument("-k", type=int, default=4, help="results to return")
    sub = p.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", parents=[common], help="retrieve context for a question")
    ask.add_argument("question")
    ask.add_argument("--prompt", action="store_true", help="print the full LLM prompt")
    ask.set_defaults(func=cmd_ask)

    idx = sub.add_parser("index", parents=[common], help="chunk documents and save the index")
    idx.add_argument("--out", required=True, help="output directory")
    idx.set_defaults(func=cmd_index)

    ev = sub.add_parser("eval", parents=[common], help="score retrievers on labelled queries")
    ev.add_argument("--queries", required=True, help="JSONL with {query, relevant: [doc_id]}")
    ev.add_argument("--compare", nargs="+", choices=RETRIEVERS, help="evaluate several retrievers")
    ev.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

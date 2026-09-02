"""Retrieval eval: Recall@k / Precision@k / MRR against the gold set.

Usage:
    python3 eval/run_eval.py --state ~/klipper-rag-state/kb.sqlite [-k 3] [--json out.json]

Gold format (eval/gold/queries.jsonl), one JSON object per line:
    {"id","category","query","gold":[[doc, section_prefix], ...],
     "must_contain": [...]}          # must_contain is answer-level (phase 4)

A retrieved chunk matches a gold entry when chunk.doc == doc AND
chunk.section startswith section_prefix ("" = any section of the doc).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kb_rag.embed import EmbedClient
from kb_rag.index import KbIndex

GOLD_PATH = Path(__file__).resolve().parent / "gold" / "queries.jsonl"


def load_gold(path: Path = GOLD_PATH) -> list[dict]:
    out = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        for field in ("id", "query", "gold", "category"):
            if field not in obj:
                raise ValueError(f"gold line {i}: missing '{field}'")
        out.append(obj)
    return out


def chunk_matches(chunk, gold_entry: list) -> bool:
    doc, prefix = gold_entry[0], gold_entry[1]
    if chunk.doc != doc:
        return False
    if not prefix:
        # whole-doc gold: every chunk except the document-title chunk
        return chunk.section != doc.replace("_", " ")
    # substring match: robust to exact heading wording ("Lost communication"
    # inside the full FAQ question title)
    return prefix.lower() in (chunk.section or "").lower()


def evaluate(gold: list[dict], index: KbIndex, client: EmbedClient, k: int,
             mode: str = "dense", sparse_weight: float = 0.5,
             rerank_url: str = "http://127.0.0.1:8101/v1/rerank"):
    per_query = []
    per_cat = defaultdict(lambda: {"n": 0, "recall": 0.0, "mrr": 0.0})
    misses = []
    for item in gold:
        qv = client.embed_query(item["query"])
        if mode in ("hybrid", "rerank"):
            from kb_rag.retrieve import hybrid_search
            if mode == "rerank":
                from kb_rag.rerank import RerankClient, rerank_hybrid
                rc = RerankClient(rerank_url)
                search_fn = lambda n, _q=item["query"]: hybrid_search(
                    index, qvec=qv, text=_q, k=n)
                results = rerank_hybrid(rc, item["query"], search_fn,
                                        k=k, candidates=10)
            else:
                results = hybrid_search(index, qvec=qv, text=item["query"],
                                        k=k, sparse_weight=sparse_weight)
        elif mode == "sparse":
            results = index.search_text(item["query"], k=k)
        else:
            results = index.search_vector(qv, k=k)
        # dedupe: any gold entry matching counts
        hit_ranks = []
        for rank, r in enumerate(results, 1):
            if any(chunk_matches(r.chunk, g) for g in item["gold"]):
                hit_ranks.append(rank)
        n_gold = len(item["gold"])
        recall = 1.0 if hit_ranks else 0.0        # recall@k: any gold found
        prec = len(hit_ranks) / k
        mrr = 1.0 / hit_ranks[0] if hit_ranks else 0.0
        per_cat[item["category"]]["n"] += 1
        per_cat[item["category"]]["recall"] += recall
        per_cat[item["category"]]["mrr"] += mrr
        if not hit_ranks:
            misses.append({
                "id": item["id"], "category": item["category"],
                "query": item["query"],
                "top": [f"{r.chunk.doc}::{r.chunk.section}" for r in results],
            })
        per_query.append({"id": item["id"], "category": item["category"],
                          "hit_ranks": hit_ranks, "recall": recall,
                          "precision": prec, "mrr": mrr})
    n = len(gold)
    overall = {
        "n": n, "k": k,
        "recall_at_k": sum(q["recall"] for q in per_query) / n,
        "precision_at_k": sum(q["precision"] for q in per_query) / n,
        "mrr": sum(q["mrr"] for q in per_query) / n,
    }
    cats = {
        c: {"n": d["n"],
            "recall_at_k": d["recall"] / d["n"],
            "mrr": d["mrr"] / d["n"]}
        for c, d in per_cat.items()
    }
    return {"overall": overall, "by_category": cats, "per_query": per_query,
            "misses": misses}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--gold", default=str(GOLD_PATH))
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--mode", choices=("dense", "sparse", "hybrid", "rerank"),
                    default="dense")
    ap.add_argument("--rerank-url",
                    default="http://127.0.0.1:8101/v1/rerank")
    ap.add_argument("--sparse-weight", type=float, default=0.5,
                    help="RRF weight of the FTS5 leg in hybrid mode")
    ap.add_argument("--embed-url", default=None)
    ap.add_argument("--json", default=None, help="write full report to file")
    args = ap.parse_args()

    index = KbIndex.load(Path(args.state))
    url = args.embed_url or index.meta.get("embed_url")
    if not url:
        print("error: no embed_url in index meta and none given", file=sys.stderr)
        return 2
    client = EmbedClient(base_url=url)

    gold = load_gold(Path(args.gold))
    report = evaluate(gold, index, client, args.k, mode=args.mode,
                      sparse_weight=args.sparse_weight,
                      rerank_url=args.rerank_url)

    o = report["overall"]
    print(f"mode={args.mode} n={o['n']} k={o['k']}  "
          f"Recall@{o['k']}={o['recall_at_k']:.3f} "
          f"P@{o['k']}={o['precision_at_k']:.3f} MRR={o['mrr']:.3f}")
    for cat, s in sorted(report["by_category"].items()):
        print(f"  {cat:16} n={s['n']:2}  recall={s['recall_at_k']:.3f}  mrr={s['mrr']:.3f}")
    if report["misses"]:
        print(f"\nmisses ({len(report['misses'])}):")
        for m in report["misses"]:
            print(f"  [{m['category']}] {m['id']}: {m['query']}")
            print(f"      got: {m['top']}")
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
    # exit non-zero if recall below the phase-1 dense gate (80%; hybrid
    # gate is 90% in the plan) — CI-able later
    return 0 if o["recall_at_k"] >= 0.80 else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Rerank eval with latency measurement: reranks hybrid top-10 -> k."""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path.home() / "klipper-rag"))
sys.path.insert(0, str(Path.home() / "klipper-rag/eval"))

from run_eval import GOLD_PATH, chunk_matches, load_gold  # noqa: E402
from kb_rag.embed import EmbedClient  # noqa: E402
from kb_rag.index import KbIndex  # noqa: E402
from kb_rag.rerank import RerankClient, rerank_hybrid  # noqa: E402
from kb_rag.retrieve import hybrid_search  # noqa: E402

K = 3
CANDIDATES = 10

idx = KbIndex.load(Path.home() / "klipper-rag-state/kb.sqlite")
ec = EmbedClient(base_url=idx.meta["embed_url"])
rc = RerankClient("http://127.0.0.1:8101/v1/rerank")

gold = load_gold(GOLD_PATH)
lat = []
per_cat = {}
misses = []
for item in gold:
    qv = ec.embed_query(item["query"])
    search_fn = lambda n, _q=item["query"]: hybrid_search(  # noqa: E731
        idx, qvec=qv, text=_q, k=n)
    t0 = time.perf_counter()
    results = rerank_hybrid(rc, item["query"], search_fn, k=K,
                            candidates=CANDIDATES)
    lat.append((time.perf_counter() - t0) * 1000)
    hit = any(any(chunk_matches(r.chunk, g) for g in item["gold"])
              for r in results)
    mrr = 0.0
    for rank, r in enumerate(results, 1):
        if any(chunk_matches(r.chunk, g) for g in item["gold"]):
            mrr = 1.0 / rank
            break
    d = per_cat.setdefault(item["category"], [0, 0.0, 0.0])
    d[0] += 1
    d[1] += 1.0 if hit else 0.0
    d[2] += mrr
    if not hit:
        misses.append(item["id"])

n = len(gold)
print(f"mode=rerank n={n} k={K}  Recall@{K}={sum(v[1] for v in per_cat.values())/n:.3f} "
      f"MRR={sum(v[2] for v in per_cat.values())/n:.3f}")
for c, (cn, r, m) in sorted(per_cat.items()):
    print(f"  {c:16} n={cn:2}  recall={r/cn:.3f}  mrr={m/cn:.3f}")
lat.sort()
print(f"retrieval+rerank latency ms: p50={lat[len(lat)//2]:.0f} "
      f"p90={lat[int(len(lat)*0.9)]:.0f} max={lat[-1]:.0f}")
print(f"misses ({len(misses)}): {misses}")

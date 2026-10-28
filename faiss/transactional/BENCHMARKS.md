# Development performance qualification

These are synthetic development measurements, not a customer latency or recall
SLO. The benchmark is intended to make the storage/recall tradeoff visible and
repeatable before integrating a production workload.

## Reproduction

Build with `FAISS_TRANSACTIONAL_BENCHMARK=ON` as described in [README.md](README.md).
Run with one OpenMP/BLAS thread, for example:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  build/faiss/transactional/faiss_transactional_benchmark \
  --n 10000 --d 128 --queries 100 --k 10 --batch 128 \
  --m 32 --ef-construction 200 --ef 64 --seed 12345 \
  --metric l2 --selectivity 1 --churn 0 --mode ann
```

Vary `--d`, `--selectivity`, `--ef`, `--mode ann|acorn`, `--churn`, and workload
size. JSON output records the selected options, exact-ground-truth recall,
returned-neighbor count, latency distribution, construction/commit time, writer
duration, record-page high-water mark, and search work counters. Result checking
rejects duplicate or ineligible IDs. Queries are stable across churn settings.

ACORN mode uses up to ten deterministic matching seeds selected from the known
predicate membership. The application must provide equivalent seeds; that
predicate evaluation/seed discovery cost is outside the timed search. The native
comparison uses ordinary filtered `IndexHNSWFlat`, not ACORN. Equal `ef_search`
is a common configuration, not a matched-recall comparison.

## Scope of the measurements

The development host is an Intel Core i7-11850H (8 cores/16 threads), x86-64,
using Ubuntu GCC 15.2 and a Release (`-O3 -DNDEBUG`) build with generic FAISS
kernels. Both engines use the same FAISS distance kernels, random seed, vectors,
queries, M and construction/search breadth. Their graph construction algorithms
and topology differ. Runtime SIMD-dispatch builds are checked separately; these
numbers do not describe their latency.

Queries run sequentially in one warm read snapshot after 16 warmups. Timed order
alternates between engines. Exact-ground-truth computation, membership/seed
selection, and index construction are outside query timings. LMDB uses normal
synchronous commits and a database directory under `/tmp`; it is already resident
from construction. Source vectors and the exact and native indexes remain
resident, so the process has more memory pressure than a standalone LMDB server.

One hundred queries are insufficient for reliable service tail-latency SLOs.
These runs do not qualify cold storage, million-vector sets, concurrent clients,
writer contention, or representative customer embeddings. Before rollout,
sweep `ef_search` at matched recall on customer data and measure p95/p99 under
expected concurrency, predicate selectivity, tombstone density and memory limits.

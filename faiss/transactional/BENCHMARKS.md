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

Vary `--d`, `--selectivity`, `--ef`, `--mode ann|acorn|exact`, `--churn`, and workload
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

## Results: 10,000 vectors, October 27, 2026

The pinned engine is `71af09b499d1fd1bb6d3420ca1457270a958df63`. Raw JSON and
[environment/source hashes](benchmarks/2026-10-27/environment.json) accompany
these results. All rows use squared L2, k=10, M=32, efConstruction=200 and
efSearch=64. The 1% predicate selected 99 vectors.

| Dimensions | Predicate | LMDB traversal | LMDB recall@10 | LMDB mean / p95 ms | Native recall@10 | Native mean / p95 ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 128 | all | [ANN](benchmarks/2026-10-27/final-d128-s1-ann.json) | 0.928 | 1.472 / 1.625 | 0.924 | 0.150 / 0.195 |
| 768 | all | [ANN](benchmarks/2026-10-27/final-d768-s1-ann.json) | 0.834 | 3.611 / 3.873 | 0.832 | 0.563 / 0.682 |
| 128 | 1% | [ANN](benchmarks/2026-10-27/final-d128-s0.01-ann.json) | 0.585 | 1.514 / 1.740 | 0.582 | 0.154 / 0.221 |
| 768 | 1% | [ANN](benchmarks/2026-10-27/final-d768-s0.01-ann.json) | 0.527 | 3.620 / 3.826 | 0.497 | 0.567 / 0.701 |
| 128 | 1% | [ACORN](benchmarks/2026-10-27/final-d128-s0.01-acorn.json) | 0.999 | 4.093 / 4.282 | 0.582 | 0.170 / 0.223 |
| 768 | 1% | [ACORN](benchmarks/2026-10-27/final-d768-s0.01-acorn.json) | 0.995 | 4.863 / 5.733 | 0.497 | 0.617 / 1.023 |

Unfiltered LMDB search is about 9.8x slower at 128 dimensions and 6.4x slower at
768 dimensions than native FAISS at similar recall in this setup. Moving graph
and vector access into LMDB has a substantial query cost; the current engine
should not be described as matching native in-memory latency. Ordinary bounded
filtered search preserves that latency shape but loses recall for selective
predicates. ACORN raises filtered recall while doing more adjacency work.

Two additional 128-dimensional, 1% predicate runs illustrate why query planning
matters:

| LMDB mode | efSearch | LMDB recall@10 | LMDB mean / p95 ms | Native ANN recall@10 | Native ANN mean / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Exact scan](benchmarks/2026-10-27/final-d128-s0.01-exact.json) | 64 | 1.000 | 0.052 / 0.059 | 0.582 | 0.095 / 0.108 |
| [ACORN](benchmarks/2026-10-27/final-d128-s0.01-acorn-ef512.json) | 512 | 0.999 | 5.178 / 5.429 | 0.973 | 0.768 / 0.928 |

The exact mode tests all 10,000 local IDs against the predicate but reads only
99 vectors and no adjacency records. It is the fastest LMDB path here and gives
complete recall. This does not establish an advantage over native exact filtered
search: the native comparator in this table is still HNSW. Exact mode ignores
`efSearch`. The ef=512 row also remains an unequal-recall comparison; it is one
point on a curve, not a matched-recall qualification.

Native latency varies across runs despite identical native parameters. These
are single short runs with both engines sharing a process and cache hierarchy;
interpret the differences as indicative, not stable service-level ratios.

### Synchronous write cost

The unfiltered builds used batches of 128, totaling 79 write transactions.

| Dimensions | LMDB preparation + child apply | Parent commits | Native construction | Longest writer ownership | LMDB file |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 26.65 s | 0.41 s | 1.66 s | 597.6 ms | 14.98 MiB |
| 768 | 69.71 s | 0.44 s | 7.52 s | 1535.0 ms | 47.32 MiB |

The writer durations are observed ownership times, not measurements of unrelated
writers waiting in a queue. Although reads retain their MVCC snapshots, write
batches share LMDB's single writer with application graph updates. Production
batch sizes must therefore account for graph-write latency as well as index
throughput.

The pinned implementation passed all 56 tests in static, shared runtime-dispatch,
and fully ASan/UBSan-instrumented CI builds, plus relocated installed-package
consumers and a core-only build without LMDB. See the
[successful CI run](https://github.com/aperture-data/faiss/actions/runs/40609699734).
This validates the tested contracts; it does not establish customer-scale
performance or complete Athena's runtime/changelog integration.

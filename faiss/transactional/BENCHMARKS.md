# Packed transactional HNSW performance

Packed v2 removes most of the original 6–10x unfiltered search penalty while
keeping vectors, adjacency, tombstones and application graph changes in one
LMDB transaction. It does **not** establish native-latency parity or a production
SLO. The measurements below are warm, single-thread synthetic workloads.

The query implementation was published in `73e531960`, with test/contract
refinements in `316297b53`. The final implementation pin is
`df147a72fddeafce7b5c5938e8b846a34d3a85d8`, which additionally fixes administrative
packed-block validation without changing search or construction. Raw results and
[environment/source provenance](benchmarks/2026-10-27-packed/environment.json)
are archived alongside this report. The original record-per-vector baseline is
retained below for comparison and is no longer the recommended query layout.

## Method and reproduction

Hardware: Intel Core i7-11850H, x86-64, GCC 15.2, Release `-O3 -DNDEBUG`, one
OpenMP/BLAS thread. Generic and runtime-dispatch (DD, AVX512_VPOPCNT on this host)
builds compare each engine with the same FAISS kernels. There is no concurrent
compilation during timed runs. Both indexes, source vectors and exact ground
truth are resident. One read transaction supplies all LMDB queries; operation
caches are freshly created for each query. Timed engine order alternates after
16 warmups. There are 200 queries per row, k=10, squared L2, M=32,
efConstruction=200 and efSearch=64 unless stated otherwise.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  build/faiss/transactional/faiss_transactional_benchmark \
  --n 10000 --d 128 --queries 200 --warmup 16 --k 10 --batch 128 \
  --m 32 --ef-construction 200 --ef 64 --seed 12345 \
  --metric l2 --layout packed --selectivity 1 --churn 0 --mode ann
```

Use `-DFAISS_OPT_LEVEL=generic` or `dd` when building. Add `--churn 0.1` for
replacement tests; set `--selectivity 0.01 --mode exact` or `acorn` for sparse
filters. For the 100k query-only fixture, use `--n 100000 --build native-materialized`.
This builds native FAISS once, imports its exact topology into LMDB and validates
it. Its materialization time is **not** transactional HNSW ingestion throughput.
The ordinary 10k rows build each engine through its own insertion path.

Equal efSearch is a common configuration, not a matched-recall qualification.
Even on the same topology, upper-routing result collection and floating-point
order can produce small differences. Inspect both recall columns. These short
runs do not establish cold-cache, concurrent-client, writer-contention or customer
embedding performance; p95/p99 are especially sensitive to host noise.

## Unfiltered search

| Case | Recall@10 packed / native | Mean ms packed / native | p95 ms packed / native | Mean ratio |
| --- | ---: | ---: | ---: | ---: |
| [generic, 10k × 128d](benchmarks/2026-10-27-packed/generic-128.json) | 0.9230 / 0.9210 | 0.229 / 0.122 | 0.290 / 0.170 | 1.88x |
| [dd, 10k × 128d](benchmarks/2026-10-27-packed/dd-128.json) | 0.9230 / 0.9210 | 0.215 / 0.103 | 0.267 / 0.150 | 2.09x |
| [generic, 10k × 768d](benchmarks/2026-10-27-packed/generic-768.json) | 0.8520 / 0.8425 | 0.798 / 0.626 | 1.393 / 1.275 | 1.27x |
| [dd, 10k × 768d](benchmarks/2026-10-27-packed/dd-768.json) | 0.8520 / 0.8425 | 0.831 / 0.621 | 1.644 / 1.357 | 1.34x |

At 10k, the remaining mean overhead is approximately 1.9–2.1x at 128d and
1.3x at 768d, with comparable measured recall. The v2 path replaces roughly
2,660/2,794 per-vector B-tree lookups with about 40/157 vector-block lookups per
query. SIMD kernels read mapped little-endian floats directly; batched distances,
compact traversal scratch and the native bounded candidate heap remove further
allocation and bookkeeping overhead. Graph nodes still use checked owned
record decoding, accounting for part of the remaining gap.

## Deletes and updates

Ten percent churn replaces 1,000 vectors; both engines retain old nodes as
excluded routing bridges. Packed storage checks transactional tombstone bits
without fetching a graph node for each candidate.

| Case | Recall@10 packed / native | Mean ms packed / native | p95 ms packed / native | Mean ratio |
| --- | ---: | ---: | ---: | ---: |
| [generic, 10k × 128d, 10% churn](benchmarks/2026-10-27-packed/generic-128-churn10.json) | 0.9225 / 0.9195 | 0.220 / 0.124 | 0.291 / 0.164 | 1.78x |
| [dd, 10k × 128d, 10% churn](benchmarks/2026-10-27-packed/dd-128-churn10.json) | 0.9225 / 0.9195 | 0.258 / 0.146 | 0.401 / 0.273 | 1.77x |
| [generic, 10k × 768d, 10% churn](benchmarks/2026-10-27-packed/generic-768-churn10.json) | 0.8320 / 0.8250 | 0.909 / 0.768 | 1.593 / 1.422 | 1.18x |
| [dd, 10k × 768d, 10% churn](benchmarks/2026-10-27-packed/dd-768-churn10.json) | 0.8320 / 0.8255 | 0.763 / 0.617 | 1.193 / 1.139 | 1.24x |

Each churn row reads only three bitmap blocks per query and about 67 graph
nodes, preserving the improvement after replacements. Deletes do not rewrite
vector blocks. This is moderate churn, not a qualification of indefinite
append-only updates; generation rebuild/retention remains an application task.

## Same graph at 100,000 vectors

| Case | Recall@10 packed / native | Mean ms packed / native | p95 ms packed / native | Mean ratio |
| --- | ---: | ---: | ---: | ---: |
| [generic, 100k × 128d, same graph](benchmarks/2026-10-27-packed/generic-100k-128-native-fixture.json) | 0.6430 / 0.6430 | 0.644 / 0.424 | 1.174 / 0.969 | 1.52x |
| [dd, 100k × 128d, same graph](benchmarks/2026-10-27-packed/dd-100k-128-native-fixture.json) | 0.6435 / 0.6435 | 0.534 / 0.296 | 0.631 / 0.379 | 1.80x |
| [generic, 100k × 768d, same graph](benchmarks/2026-10-27-packed/generic-100k-768-native-fixture.json) | 0.5035 / 0.5035 | 1.351 / 0.820 | 1.514 / 0.942 | 1.65x |
| [dd, 100k × 768d, same graph](benchmarks/2026-10-27-packed/dd-100k-768-native-fixture.json) | 0.5010 / 0.5010 | 1.266 / 0.732 | 1.418 / 0.869 | 1.73x |

These rows isolate representation/traversal overhead from graph construction.
The low recall at efSearch=64 on high-dimensional normal random vectors is not a
recommended operating point. Customer qualification needs an efSearch sweep at
the required recall, using actual embeddings and concurrency.

## Selective filters

For the 1% predicate (99 eligible vectors), exact packed search takes about
0.04 ms with 100% recall and no graph-node reads. It still tests all 10,000 IDs
against the predicate, then evaluates only matching vectors. ACORN reaches
99.8% recall but takes about 2 ms and reads about 2,232 graph nodes. The native
reference at efSearch=64 reaches only 57.55% recall, so its latency is not an
apples-to-apples alternative to those high-recall results.

| Case | Recall@10 packed / native | Mean ms packed / native | p95 ms packed / native | Mean ratio |
| --- | ---: | ---: | ---: | ---: |
| [generic, 10k × 128d, 1%, exact](benchmarks/2026-10-27-packed/generic-128-exact1pct.json) | 1.0000 / 0.5755 | 0.039 / 0.092 | 0.041 / 0.105 | 0.42x |
| [dd, 10k × 128d, 1%, exact](benchmarks/2026-10-27-packed/dd-128-exact1pct.json) | 1.0000 / 0.5755 | 0.037 / 0.083 | 0.039 / 0.091 | 0.45x |
| [generic, 10k × 128d, 1%, acorn](benchmarks/2026-10-27-packed/generic-128-acorn1pct.json) | 0.9980 / 0.5755 | 2.030 / 0.154 | 2.146 / 0.232 | 13.18x |
| [dd, 10k × 128d, 1%, acorn](benchmarks/2026-10-27-packed/dd-128-acorn1pct.json) | 0.9980 / 0.5755 | 2.051 / 0.137 | 2.150 / 0.210 | 15.02x |

The ratios in this filter table describe measured modes with different recall;
they are not storage-overhead estimates. The native comparator is ordinary
filtered HNSW, even when the packed mode is exact or ACORN. ACORN uses up to ten
deterministic matching seeds; predicate/seed discovery is outside timed search.
For small eligible sets, explicitly select exact search. Larger selective
workloads still need query planning and adjacency-path optimization.

Higher-breadth same-graph fixtures provide a closer recall comparison:

| Case | Recall@10 packed / native | Mean ms packed / native | p95 ms packed / native | Mean ratio |
| --- | ---: | ---: | ---: | ---: |
| [DD, same graph, 1%, ACORN, ef=512](benchmarks/2026-10-27-packed/dd-10k-128-acorn1pct-ef512-native-fixture.json) | 1.0000 / 0.9810 | 2.533 / 0.469 | 2.679 / 0.616 | 5.40x |
| [DD, same graph, 1%, ACORN, ef=1024](benchmarks/2026-10-27-packed/dd-10k-128-acorn1pct-ef1024-native-fixture.json) | 1.0000 / 0.9975 | 2.561 / 0.781 | 2.694 / 0.991 | 3.28x |

At efSearch=1024, the remaining ACORN gap is about **3.3x** at near-equal
recall (100% versus 99.75%). That path is not close to native latency;
small eligible sets should use the measured exact-search alternative.

## Synchronous write cost and validation

Packed blocks trade write amplification for fast reads. In the DD unfiltered
10k builds, batches of 128 took 19.31 s plus 0.41 s commits at 128d and 40.45 s
plus 0.44 s commits at 768d; native construction took 1.43 s and 5.25 s. Maximum
observed writer ownership was 446 ms and 949 ms respectively. These writer holds
share LMDB's single writer with application graph writes, so production batch
sizes must account for graph-write latency. LMDB files were 13.52 MiB and
38.64 MiB. Query-only fixture materialization is not a substitute for these
insertion measurements.

The final revision passed all **92 tests** in static generic, shared runtime-
dispatch and fully ASAN/UBSAN-instrumented CI, plus installed-package consumers
and the core-only build without LMDB. See the
[successful CI run](https://github.com/aperture-data/faiss/actions/runs/40617243736).

Correctness coverage includes both record formats, old-reader snapshots,
transaction/savepoint rollback, map-full failures after partial writes, callback
replay, bitmap consistency, packed block corruption, budgets, cancellation,
filters, batch distances and operation-local cache lifetimes. Installed static
and shared consumers exercise packed create/commit/reopen/search. The fork adds
an opt-in C++ backend; the Athena dependency pin does not enable its query routing
or complete transaction/changelog integration. Consumers must rebuild when
upgrading the transactional API.

## Historical record-per-vector baseline

The following measurements describe the older `71af09b4` implementation only.
They are retained as evidence of the initial problem, not as current packed-v2
performance figures.


These are synthetic development measurements, not a customer latency or recall
SLO. The benchmark is intended to make the storage/recall tradeoff visible and
repeatable before integrating a production workload.

### Reproduction

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

### Scope of the measurements

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

### Results: 10,000 vectors, October 27, 2026

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
and vector access into LMDB has a substantial query cost; that earlier engine
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

#### Synchronous write cost

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

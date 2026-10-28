# Transaction-scoped HNSW

This Aperture Data extension adds `faiss_transactional` and `faiss_lmdb` alongside
upstream FAISS 1.15.1. It stores vectors, adjacency lists, tombstones and index
metadata in the application's LMDB environment, so one transaction can commit
application graph changes and HNSW changes together. It has no asynchronous
index updater or mutable process-global graph.

This is an opt-in C++ storage engine. Existing `IndexHNSWFlat` behavior and FAISS
serialization remain unchanged; installing this fork does not switch Athena's
existing descriptor queries to the new backend. Application integration and
workload qualification are still required before enabling it in production.

The engine adapts HNSW candidate traversal and reuses FAISS neighbor
diversification and distance kernels. Construction is serial and incremental;
it does not reproduce upstream's parallel deterministic batch construction or
promise identical topology. Supported metrics are squared L2 and inner product.
For cosine similarity, normalize vectors and queries before using inner product.
Non-finite input and non-finite computed distances are rejected.

## Build, install and test

Install a C++20 compiler, CMake, BLAS/LAPACK, OpenMP, LMDB development files, and
GTest development files. On Ubuntu the relevant packages include
`build-essential cmake ninja-build libopenblas-dev liblmdb-dev libgtest-dev`.

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
  -DFAISS_ENABLE_EXTRAS=OFF -DBUILD_TESTING=OFF \
  -DFAISS_ENABLE_TRANSACTIONAL_HNSW=ON -DFAISS_ENABLE_LMDB=ON \
  -DFAISS_TRANSACTIONAL_TESTING=ON -DFAISS_TRANSACTIONAL_BENCHMARK=ON
cmake --build build --target faiss_transactional_test faiss_transactional_benchmark -j 4
OMP_NUM_THREADS=1 ctest --test-dir build --output-on-failure
cmake --install build
```

Consumers use `find_package(faiss CONFIG REQUIRED)` and link `faiss_lmdb`.
Applications implementing their own `Storage` need only `faiss_transactional`.
LMDB is not required when `FAISS_ENABLE_LMDB=OFF`. The installed package discovers
its exported dependencies. CI checks static/shared builds, runtime SIMD dispatch,
AddressSanitizer/UndefinedBehaviorSanitizer, and a relocated installed consumer.

## Transaction ownership and atomic mutations

Open an ordinary byte-key, non-DUPSORT LMDB table in the **same environment** as
application data. The caller owns its environment, DBI, and transactions. An
adapter belongs to one transaction and its construction thread. Destroy every
adapter before committing, aborting, resetting or renewing its transaction.
Thread checks cannot detect an externally ended transaction. Do not share a
transaction with OpenMP workers or operate on the parent while a child is active.

```cpp
#include <faiss/transactional/HNSW.h>
#include <faiss/transactional/LmdbStorage.h>

namespace th = faiss::transactional;

// txn is a caller-owned write transaction; table is an ordinary byte-key DBI.
{
    th::LmdbStorage storage(txn, table, index_id, generation);
    th::HNSW index(storage);
    th::Config config;
    config.dimensions = dimensions;
    index.create(config); // only for a new namespace

    th::MutationOptions limits;
    limits.max_batch_size = 256;
    auto ids = index.add(count, vectors, limits);
    // Write application entity -> local ID mappings in txn as well.
    auto result = index.search(query, 10); // sees pending additions
}
// Commit application data and index changes together, or abort both.
```

The same API reads a caller-owned read transaction. Entry point, vectors, edges,
and tombstones all come from that snapshot, even if a writer commits concurrently.
No borrowed LMDB value pointers escape the public adapter API. Read methods
decode directly from the mapped record into owned results before returning;
operation-local caches never survive the operation or transaction.

`add`, `update`, and `erase` prepare all changes before writing. Input errors,
cancellation, traversal limits and workspace limits during preparation leave
storage unchanged. `update(old_id, vector)` atomically tombstones the old ID and
inserts a replacement, returning a **new** ID; update the application mapping in
the same transaction. Deleting an already deleted/unknown ID returns false.

The default callback-free LMDB adapter applies each batch in a child transaction.
This guarantee covers HNSW operations and `Storage::apply`; the adapter's low-level
`put_*` methods write individual records and do not preserve graph consistency
when called independently.
A successful child commit only merges changes into the parent; nothing becomes
visible to other readers until the caller commits the parent. Write failure
aborts the child and invalidates the adapter. Catch `MutationError` and inspect
`abort_required()`:

| Failure | Caller action |
| --- | --- |
| Preparation failed before storage writes | The transaction and adapter remain usable. |
| `MutationError(false)` | Discard the adapter. The batch was rolled back; prior parent changes remain. A fresh adapter can continue. |
| `MutationError(true)` | Discard the adapter and abort the entire parent transaction; discard associated external changelog buffers. |
| Caller commit fails | Handle according to LMDB's transaction ownership/error rules. |

The original storage error is nested in `MutationError`; `LmdbError::code()`
preserves `MDB_MAP_FULL` and other LMDB error codes. A child commit failure
conservatively requires parent abort. Do not resize an LMDB mapping while active
transactions use it; perform recovery at the application's environment boundary.

`WriteMode::CallerTransaction` writes directly to the parent and requires abort
on any apply failure. It is necessary for environments using `MDB_WRITEMAP`,
which cannot create child transactions. Automatic mode rejects such writes
before modifying records; explicitly select caller mode if needed.

## Search latency, filtering and recall

```cpp
th::SearchOptions options;
options.ef_search = 128;
options.max_visits = 20000;
options.selector = allowed_local_ids; // optional faiss::IDSelector
th::SearchStats stats;
auto results = index.search(query, 10, options, &stats);
if (stats.visit_limit_reached) {
    // Apply the application's partial-result or retry policy.
}
```

Results are ordered by increasing squared L2 or decreasing inner product, with
local-ID tie-breaking. Tombstones and predicate-rejected nodes are excluded from
results but remain routing bridges. The ordinary search exploration heap is
independent of the filter: a selective predicate cannot force the engine to scan
the whole graph merely to fill `ef_search` accepted candidates.

`ef_search` controls the graph-search breadth/recall tradeoff. `max_visits` limits
unique query distance evaluations, including upper levels and explicit seeds;
zero derives a finite bound from `ef_search`, M and graph height. Exhaustion
returns the best accepted results found and sets `visit_limit_reached`. The
bound is on distance evaluations, not elapsed time or record reads. Search can
return fewer than k neighbors even without hitting the limit: HNSW is approximate
and graph components can be disconnected, especially with duplicate vectors.

For selective predicates, set `acorn=true` to explore accepted one/two-hop
neighbors through rejected bridge nodes without evaluating those bridges'
vectors at level zero. Provide up to 64 snapshot-local `seed_ids` from the
application's filtered graph candidates to reach distant matching regions.
This is an ACORN-style traversal option, not a port of Athena's existing ACORN
integration or a promise of identical results. Upper-level routing still uses
ordinary distances. Benchmark its adjacency-read cost as well as recall; sparse
predicates are not automatically cheap.

For guaranteed top-k across disconnected components, set `exact=true` and leave
`max_visits=0`. This streams the snapshot's IDs and maintains a k-sized result
heap without storing all visited distances or vectors. An explicit nonzero
limit can truncate even exact mode; inspect `visit_limit_reached`. Exact search
is a deliberate full scan, useful for ground truth or small filtered sets, and
is not an automatic fallback hidden inside ANN search.

Search statistics expose distance evaluations, node/vector storage reads,
adjacency expansions (including two-hop bridges), and budget exhaustion.
`cache_bytes` bounds owned decoded-record caches; search does not retain decoded
vectors because query distances are computed once. Distance/visited bookkeeping
is proportional to the visit limit, separately from the cache allowance. ACORN
also inspects bounded-degree bridge adjacency. Exact scanning uses bounded
cache and result memory rather than memory proportional to index cardinality.

Cancellation callbacks are polled throughout traversal and mutation preparation
and throw `SearchCancelled`. Selector and cancellation callbacks must not mutate
the transaction or reenter the index; referenced state must outlive the call.
Batch flushes are not interrupted halfway by cancellation.

## Writes, capacity and retention

`MutationOptions` bounds batch size (default 10,000 vectors), buffered dirty
records (default 256 MiB), read caches (default 8 MiB), and unique distances from
each inserted vector to graph candidates (zero derives a finite construction
bound). Neighbor diversification performs additional pairwise distances. The workspace value
accounts for dirty payloads and estimated container overhead, not a strict
process-RSS quota; traversal bookkeeping and bounded transient records are
additional. Exceeding a preparation limit fails the entire operation before any
records are applied. Split large imports into explicit batches/transactions.

Local IDs are signed 32-bit, append-only, and scoped to an index generation.
All IDs from zero through `INT32_MAX` can be allocated. Deletes leave vectors
and edges in place for routing; updates consume new IDs. Rebuild into a new
generation to reclaim tombstones and reset capacity. Online rebuild/catch-up,
generation switching, and garbage collection belong to the application.
Synchronous insertion holds the shared LMDB writer, so bound write transactions
and measure writer occupancy alongside search latency.

## Record format and replication

One table stores keys prefixed by `TH01`, a 64-bit index ID and a 64-bit generation.
Metadata (`m`), node (`n` plus local ID), and vector (`v` plus local ID) records
are separate. Node records contain all levels; changing links does not rewrite
the vector. Integers and IEEE-754 float bits are big-endian. Metadata includes
format version 1. Checked decoders reject malformed sizes, invalid numeric
values, invalid/duplicate neighbor IDs, and excessive degrees/levels. Use a new
namespace/generation for incompatible formats.

`LmdbStorage::Put` integrates with a database changelog. It receives the existing
transaction, DBI, and temporary encoded key/value views. The callback must apply
the write to that **same** transaction before returning and copy the bytes it
needs for replication. Replicate the resulting records rather than rerunning
HNSW construction independently on replicas. The callback must not reenter the
adapter or perform index operations during a partially applied batch.

Supplying a callback selects caller-transaction mode automatically. External
changelog side effects cannot be rolled back by an LMDB child transaction, so
callback plus explicit savepoint mode is rejected. On any batch-write failure,
abort the parent and discard its external changelog buffers. A newly constructed
adapter cannot discover that an earlier caller-mode batch failed: enforcing the
abort contract remains the transaction owner's responsibility.

## Validation and rollout

`validate()` performs an administrative full scan of vectors, counts, degrees,
IDs, cross-level edges and entry-point consistency. It does not require graph
connectivity. Run it during migration verification or offline diagnosis, not
before every query.

The tests cover commit/reopen, application-write rollback, old-reader snapshots,
failed updates, map-full savepoint recovery, callback failure/reentry, changelog
replay, thread ownership, corruption, ID boundaries, cancellation, bounded work,
selective filters, cache eviction, duplicates, L2/IP/cosine recall, and unchanged
upstream `IndexHNSWFlat`. They are correctness checks, not customer SLO proof.

The benchmark emits JSON with exact-ground-truth recall, query p50/p95/p99,
construction/commit times, writer duration, storage sizes, and traversal counts.
For example:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  build/faiss/transactional/faiss_transactional_benchmark \
  --n 10000 --d 128 --queries 200 --ef 128 --batch 100 \
  --selectivity 0.01 --mode acorn > benchmark.json
```

Benchmark real cardinalities, dimensions, metrics, selectivity, churn, warm/cold
cache, concurrent readers and writer contention before rollout. The included
benchmark measures warm-cache queries in one process; native FAISS is a separate
in-memory topology and is a reference, not a transactional substitute. See
[BENCHMARKS.md](BENCHMARKS.md) for the measured development baseline.

Athena still needs transaction-scoped routing, entity/local-ID mappings,
normalization rules, changelog integration, predicate/seed translation,
existing-index migration, generation retention and operational monitoring.
Its current `DbMapFullChangelog::put` can suppress `MDB_MAP_FULL` and stop applying
writes while still logging them. It therefore cannot be passed directly as this
adapter's `Put` callback, which requires successful application before return.
Athena also defers graph mutations until changelog construction. Integration must
provide a transaction-local pending-record overlay and emit its final writes at
commit, or explicitly redesign that deferred/retry path; a direct callback would
break read-your-writes and error handling.
The dependency pin makes the backend available without changing existing query
routing.

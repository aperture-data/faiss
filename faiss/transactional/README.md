# Transaction-scoped HNSW

This Aperture Data extension adds `faiss_transactional` and `faiss_lmdb` alongside
upstream FAISS 1.15.1. Existing `IndexHNSWFlat` behavior and serialization are
unchanged. The extension is an experimental, opt-in storage engine, not a
transparent replacement for the existing FAISS index classes.

The engine adapts HNSW candidate traversal and reuses FAISS neighbor
diversification and distance kernels. Construction is serial and incremental;
it does not reproduce upstream's parallel deterministic batch construction or
promise identical topology. Local IDs are signed 32-bit, append-only, and scoped
to an index generation. Cosine similarity is supported by supplying normalized
vectors and queries with `METRIC_INNER_PRODUCT`; normalization is the caller's
responsibility. Non-finite input and non-finite computed distances are rejected.

## Build and test

Install a C++20 compiler, CMake, BLAS/LAPACK, OpenMP, LMDB development files, and
GTest development files. For example, on Ubuntu install `build-essential cmake
ninja-build libopenblas-dev liblmdb-dev libgtest-dev`.

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
  -DFAISS_ENABLE_EXTRAS=OFF -DBUILD_TESTING=OFF \
  -DFAISS_ENABLE_TRANSACTIONAL_HNSW=ON -DFAISS_ENABLE_LMDB=ON \
  -DFAISS_TRANSACTIONAL_TESTING=ON
cmake --build build --target faiss_transactional_test -j 4
OMP_NUM_THREADS=1 ctest --test-dir build --output-on-failure
cmake --install build
```

Link applications with the exported `faiss_lmdb` target. Applications supplying
another implementation of `Storage` need only `faiss_transactional`. Neither
engine starts background workers. Ordinary FAISS builds have no LMDB dependency
unless `FAISS_ENABLE_LMDB` is enabled.

## Transaction ownership

Open an ordinary named LMDB table in the SAME environment as application data.
The caller owns its environment, DBI, and transactions. Each storage adapter and
HNSW view belongs to exactly one transaction and must be destroyed before that
transaction ends. Do not reuse adapters or cached records across snapshots.

```cpp
#include <faiss/transactional/HNSW.h>
#include <faiss/transactional/LmdbStorage.h>

// txn: caller-owned write transaction; table: ordinary byte-key DBI.
{
    faiss::transactional::LmdbStorage storage(txn, table, index_id, generation);
    faiss::transactional::HNSW index(storage);
    // Only for a new namespace:
    faiss::transactional::Config config;
    config.dimensions = dimensions;
    index.create(config);
    auto ids = index.add(count, vectors);
    // Store application entity-to-local-ID mappings using the same txn.
    auto neighbors = index.search(query, 10); // sees the pending additions
}
// Caller commits all application and index changes together, or aborts all.
```

The read path uses the same API with a caller-owned read transaction. Readers see
that transaction's entry point, vectors, topology, and tombstones, including when
another transaction commits concurrently. Search accepts an optional FAISS
`IDSelector` over local IDs; excluded and deleted nodes remain traversal bridges.
This is ordinary filtered HNSW search, not Athena's ACORN implementation.

Insertion buffers only changed nodes and new vectors before flushing. If flushing
fails (including `MDB_MAP_FULL` or a changelog callback error), the adapter is
invalidated. **The caller must abort the entire transaction**, not commit a
partially flushed batch or construct another adapter to continue it. LMDB does
not automatically roll back a failed operation. Successful earlier calls in that
transaction are also discarded on abort. Commit errors remain the caller's
responsibility.

All methods are serial. Never share a write transaction with OpenMP workers.
No borrowed LMDB value pointers escape the adapter; query caches own vector
copies and are discarded after each operation.

## Records and replication

One non-DUPSORT, byte-key table holds records under a `TH01`, 64-bit index ID,
64-bit generation prefix. Metadata, node adjacency/tombstones, and vectors have
separate keys. The node record contains all of that node's levels; changing links
does not rewrite its vector. Integers and IEEE-754 float bits are encoded
big-endian with bounded, checked decoding. Metadata records carry a format
version. Applications must choose a new namespace/generation for incompatible
formats and own generation retention/retirement.

`LmdbStorage::Put` can connect mutations to a database changelog. The callback
receives the existing transaction, DBI, and encoded key/value. It must apply the
write to that same transaction before returning and capture the bytes needed for
replication. Callback views are temporary. Readers and subsequent operations
must see pending writes. Replicate the resulting records, rather than rerunning
HNSW construction independently on replicas.

Deletion marks a node dead and decrements the live count. Its vector and edges
remain to preserve traversal. Updating a vector means tombstoning its old ID,
inserting a new vector, and changing the entity mapping in the same transaction.
Rebuild into a new generation to reclaim tombstones; online rebuild/catch-up is
not provided here. Persistent IDs are not recycled within a generation.

## Validation and remaining integration

Tests cover commit/reopen, rollback with application graph writes, old-reader
snapshots, tombstones, predicate filtering, namespace isolation, invalid input,
read-only writes, injected flush failures, changelog replay, deterministic
incremental insertion, topology invariants, and recall against exact L2/IP/cosine
search. They also exercise the unchanged upstream `IndexHNSWFlat` API.

The tests are correctness checks, not production performance qualification.
Benchmark customer dimensions, cardinalities, churn, filtered queries, warm/cold
cache behavior, and writer contention before enabling this engine. Current
vector reads copy/decode data, and synchronous insertion occupies the shared
LMDB writer. Memory grows with the dirty topology/new vectors in a write batch
and visited vectors in each query; callers should bound batch sizes.

An application must still integrate transaction-scoped query routing, entity
mappings, metric eligibility/normalization, its changelog, ACORN if required,
existing-index migration, and garbage collection. Installing this fork alone
does not switch Athena's existing descriptor queries to this backend.

/* Copyright (c) Aperture Data Inc. Licensed under the root MIT LICENSE. */
// Reproducible synthetic workload, not a substitute for customer-data
// qualification.
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/transactional/LmdbStorage.h>
#include <omp.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ft = faiss::transactional;
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start)
            .count();
}
void check(int status) {
    if (status)
        throw std::runtime_error(mdb_strerror(status));
}
struct Options {
    size_t n = 10000, dimensions = 128, queries = 200, k = 10;
    size_t batch = 128, ef = 64, m = 32, ef_construction = 200, repeats = 1;
    size_t warmup = 16, seed = 12345, map_mib = 0;
    size_t max_visits = 0, cache_mib = 8, vectors_per_block = 0;
    std::string mode = "ann", layout = "records";
    std::string build_mode = "transactional";
    double selectivity = 1, churn = 0;
    std::string metric = "l2";
};
size_t integer(const std::string& value) {
    size_t result = 0;
    const auto parsed =
            std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        throw std::invalid_argument("invalid integer: " + value);
    return result;
}
double fraction(const std::string& value) {
    size_t consumed = 0;
    const double result = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(result) || result < 0 ||
        result > 1)
        throw std::invalid_argument("fraction must be in [0,1]: " + value);
    return result;
}
Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::cout
                    << "Transactional HNSW benchmark; JSON output. One OpenMP thread.\n"
                       "  --n 10000 --d 128 --queries 200 --k 10 --batch 128\n"
                       "  --ef 64 --m 32 --ef-construction 200 --seed 12345\n"
                       "  --metric l2|ip|cosine --selectivity 1 --churn 0\n"
                       "  --repeats 1 --warmup 16 --map-mib 0 (0=estimated)\n"
                       "  --mode ann|acorn|exact --max-visits 0 --cache-mib 8\n"
                       "  --layout records|packed --vectors-per-block 0 (0=automatic)\n"
                       "  --build transactional|native-materialized\n"
                       "Native-materialized copies the native graph for query-only comparisons;\n"
                       "its materialization time is not transactional HNSW insertion time.\n"
                       "Churn replaces a fraction of initial vectors before queries.\n"
                       "Both engines retain old nodes as excluded traversal bridges.\n"
                       "Search measurements use one read snapshot, with warmup.\n";
            std::exit(0);
        }
        if (++i == argc)
            throw std::invalid_argument("missing value for " + arg);
        const std::string value = argv[i];
        if (arg == "--mode")
            o.mode = value;
        else if (arg == "--build")
            o.build_mode = value;
        else if (arg == "--layout")
            o.layout = value;
        else if (arg == "--metric")
            o.metric = value;
        else if (arg == "--selectivity")
            o.selectivity = fraction(value);
        else if (arg == "--churn")
            o.churn = fraction(value);
        else {
            const size_t n = integer(value);
            if (arg == "--n")
                o.n = n;
            else if (arg == "--d")
                o.dimensions = n;
            else if (arg == "--queries")
                o.queries = n;
            else if (arg == "--k")
                o.k = n;
            else if (arg == "--batch")
                o.batch = n;
            else if (arg == "--ef")
                o.ef = n;
            else if (arg == "--m")
                o.m = n;
            else if (arg == "--ef-construction")
                o.ef_construction = n;
            else if (arg == "--repeats")
                o.repeats = n;
            else if (arg == "--warmup")
                o.warmup = n;
            else if (arg == "--seed")
                o.seed = n;
            else if (arg == "--map-mib")
                o.map_mib = n;
            else if (arg == "--max-visits")
                o.max_visits = n;
            else if (arg == "--cache-mib")
                o.cache_mib = n;
            else if (arg == "--vectors-per-block")
                o.vectors_per_block = n;
            else
                throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (!o.n || o.n > (size_t{1} << 30) || !o.dimensions ||
        o.dimensions > (1u << 20) || !o.queries || !o.k || o.k > o.n ||
        !o.batch || !o.ef || o.ef > (1u << 24) || o.m < 2 || o.m > 4096 ||
        o.ef_construction < o.m || o.ef_construction > (1u << 24) ||
        !o.repeats || o.vectors_per_block > 4096 ||
        (o.layout != "records" && o.layout != "packed") ||
        (o.build_mode != "transactional" &&
         o.build_mode != "native-materialized") ||
        (o.metric != "l2" && o.metric != "ip" && o.metric != "cosine") ||
        (o.mode != "ann" && o.mode != "acorn" && o.mode != "exact"))
        throw std::invalid_argument(
                "invalid workload configuration; see --help");
    return o;
}
size_t multiply(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::overflow_error("workload is too large");
    return a * b;
}
struct Transaction {
    MDB_txn* txn = nullptr;
    Transaction(MDB_env* env, bool read) {
        check(mdb_txn_begin(env, nullptr, read ? MDB_RDONLY : 0, &txn));
    }
    ~Transaction() {
        if (txn)
            mdb_txn_abort(txn);
    }
    double commit() {
        const auto start = Clock::now();
        check(mdb_txn_commit(std::exchange(txn, nullptr)));
        return milliseconds(start);
    }
};
struct Database {
    MDB_env* env = nullptr;
    MDB_dbi table = 0;
    std::string path;
    explicit Database(size_t map_size) {
        char pattern[] = "/tmp/faiss-hnsw-benchmark-XXXXXX";
        const auto directory = mkdtemp(pattern);
        if (!directory)
            throw std::runtime_error("cannot create benchmark directory");
        path = directory;
        try {
            check(mdb_env_create(&env));
            check(mdb_env_set_mapsize(env, map_size));
            check(mdb_env_open(env, path.c_str(), MDB_NOTLS, 0600));
            Transaction tx(env, false);
            check(mdb_dbi_open(tx.txn, nullptr, 0, &table));
            tx.commit();
        } catch (...) {
            if (env)
                mdb_env_close(env);
            std::filesystem::remove_all(path);
            throw;
        }
    }
    ~Database() {
        mdb_env_close(env);
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};
uint64_t hash(uint64_t n) {
    n += 0x9e3779b97f4a7c15ULL;
    n = (n ^ (n >> 30)) * 0xbf58476d1ce4e5b9ULL;
    n = (n ^ (n >> 27)) * 0x94d049bb133111ebULL;
    return n ^ (n >> 31);
}
void normalize(std::vector<float>& vectors, size_t dim) {
    for (size_t i = 0; i < vectors.size(); i += dim) {
        double norm = 0;
        for (size_t j = 0; j < dim; ++j)
            norm += double(vectors[i + j]) * vectors[i + j];
        if (norm == 0)
            throw std::runtime_error("zero synthetic vector");
        norm = std::sqrt(norm);
        for (size_t j = 0; j < dim; ++j)
            vectors[i + j] /= static_cast<float>(norm);
    }
}
struct Summary {
    std::vector<double> latency;
    uint64_t hits = 0, expected = 0, returned = 0;
    void add(
            const std::vector<faiss::idx_t>& labels,
            const std::unordered_set<faiss::idx_t>& truth,
            double ms,
            size_t total,
            const faiss::IDSelector& eligible) {
        std::unordered_set<faiss::idx_t> unique;
        unique.reserve(labels.size());
        for (const auto id : labels) {
            if (id == -1)
                continue; // FAISS pads missing results with -1.
            if (id < 0 || static_cast<uint64_t>(id) >= total)
                throw std::runtime_error("search returned an out-of-range ID");
            if (!eligible.is_member(id))
                throw std::runtime_error(
                        "search returned an excluded or deleted ID");
            if (!unique.insert(id).second)
                throw std::runtime_error("search returned duplicate IDs");
        }
        latency.push_back(ms);
        expected += truth.size();
        returned += unique.size();
        for (const auto id : unique)
            hits += truth.contains(id);
    }
    void print() {
        std::sort(latency.begin(), latency.end());
        const auto quantile = [&](double p) {
            return latency[std::min(
                    latency.size() - 1,
                    static_cast<size_t>(std::ceil(p * latency.size())) - 1)];
        };
        const double total =
                std::accumulate(latency.begin(), latency.end(), 0.0);
        std::cout << "{\"recall_at_k\":"
                  << (expected ? double(hits) / expected : 1.0)
                  << ",\"expected_neighbors\":" << expected
                  << ",\"returned_neighbors\":" << returned
                  << ",\"query_count\":" << latency.size()
                  << ",\"mean_ms\":" << total / latency.size()
                  << ",\"p50_ms\":" << quantile(.5)
                  << ",\"p95_ms\":" << quantile(.95)
                  << ",\"p99_ms\":" << quantile(.99)
                  << ",\"max_ms\":" << latency.back()
                  << ",\"queries_per_second\":"
                  << (total ? 1000.0 * latency.size() / total : 0) << '}';
    }
};
void run(const Options& o) {
    omp_set_num_threads(1);
    const size_t replaced = static_cast<size_t>(std::floor(o.n * o.churn));
    const size_t total = o.n + replaced;
    if (total > static_cast<size_t>(std::numeric_limits<ft::NodeId>::max()))
        throw std::invalid_argument("too many HNSW local IDs");
    const size_t record_estimate =
            multiply(o.dimensions, 4) + multiply(o.m, 20) + 1024;
    const size_t map_size = o.map_mib
            ? multiply(o.map_mib, 1u << 20)
            : std::max<size_t>(
                      1u << 30, multiply(multiply(total, record_estimate), 4));
    multiply(o.queries, o.repeats);
    std::vector<float> vectors(multiply(total, o.dimensions));
    std::vector<float> queries(multiply(o.queries, o.dimensions));
    std::mt19937_64 rng(o.seed);
    std::normal_distribution<float> normal;
    // Keep both initial vectors and queries fixed when sweeping churn. Generate
    // replacements after queries so they cannot duplicate the query RNG stream.
    // This preserves the historical zero-churn workload byte for byte.
    const size_t initial_values = multiply(o.n, o.dimensions);
    for (size_t i = 0; i < initial_values; ++i)
        vectors[i] = normal(rng);
    for (auto& value : queries)
        value = normal(rng);
    for (size_t i = initial_values; i < vectors.size(); ++i)
        vectors[i] = normal(rng);
    if (o.metric == "cosine") {
        normalize(vectors, o.dimensions);
        normalize(queries, o.dimensions);
    }
    const auto metric =
            o.metric == "l2" ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT;
    faiss::IndexHNSWFlat native(o.dimensions, o.m, metric);
    native.hnsw.rng = faiss::RandomGenerator(static_cast<int64_t>(o.seed));
    native.hnsw.efConstruction = static_cast<int>(o.ef_construction);
    native.hnsw.efSearch = static_cast<int>(o.ef);
    Database db(map_size);
    {
        Transaction tx(db.env, false);
        {
            ft::LmdbStorage storage(
                    tx.txn,
                    db.table,
                    1,
                    0,
                    {},
                    ft::LmdbStorage::WriteMode::Automatic,
                    {o.layout == "packed",
                     static_cast<uint32_t>(o.vectors_per_block)});
            ft::HNSW index(storage);
            ft::Config config;
            config.dimensions = o.dimensions;
            config.M = o.m;
            config.ef_construction = o.ef_construction;
            config.metric = metric;
            config.seed = o.seed;
            index.create(config);
        }
        tx.commit();
    }
    double lmdb_insert_ms = 0, native_insert_ms = 0, commit_ms = 0;
    double max_writer_ms = 0, materialize_ms = 0, fixture_validation_ms = 0;
    size_t transactions = 0;
    const auto build_range = [&](size_t first, size_t end, bool churn) {
        for (size_t offset = first; offset < end; offset += o.batch) {
            const auto count = std::min(o.batch, end - offset);
            auto start = Clock::now();
            native.add(count, vectors.data() + offset * o.dimensions);
            native_insert_ms += milliseconds(start);
            if (o.build_mode == "native-materialized")
                continue;
            const auto writer_start = Clock::now();
            Transaction tx(db.env, false);
            {
                ft::LmdbStorage storage(
                        tx.txn,
                        db.table,
                        1,
                        0,
                        {},
                        ft::LmdbStorage::WriteMode::Automatic,
                        {o.layout == "packed",
                         static_cast<uint32_t>(o.vectors_per_block)});
                ft::HNSW index(storage);
                start = Clock::now();
                if (churn) {
                    for (size_t i = 0; i < count; ++i)
                        index.erase(static_cast<ft::NodeId>(offset + i - o.n));
                }
                index.add(count, vectors.data() + offset * o.dimensions);
                lmdb_insert_ms += milliseconds(start);
            }
            commit_ms += tx.commit();
            max_writer_ms = std::max(max_writer_ms, milliseconds(writer_start));
            ++transactions;
        }
    };
    build_range(0, o.n, false);
    build_range(o.n, total, true);
    if (o.build_mode == "native-materialized") {
        // This fixture measures search on exactly the same topology as native.
        // Buffer only --batch records at once, and publish the completed graph
        // in one parent transaction. It is not a transactional ingestion test.
        const auto writer_start = Clock::now();
        Transaction tx(db.env, false);
        {
            ft::LmdbStorage storage(tx.txn, db.table, 1);
            ft::HNSW index(storage);
            auto metadata = index.metadata();
            metadata.next_id = total;
            metadata.live_count = o.n;
            metadata.entry_point = native.hnsw.entry_point;
            metadata.max_level = native.hnsw.max_level;
            const auto start = Clock::now();
            for (size_t offset = 0; offset < total; offset += o.batch) {
                ft::WriteBatch batch;
                batch.metadata = metadata;
                const auto end = std::min(total, offset + o.batch);
                // Storage allocates packed live bits alongside contiguous IDs.
                // The unfinished prefix stays private to this transaction.
                batch.metadata.next_id = end;
                batch.metadata.live_count = end - std::min(end, replaced);
                if (end != total) {
                    batch.metadata.entry_point = 0;
                    batch.metadata.max_level = native.hnsw.levels[0] - 1;
                }
                for (size_t id = offset; id < end; ++id) {
                    ft::Node node;
                    node.deleted = id < replaced;
                    node.links.resize(native.hnsw.levels[id]);
                    for (size_t level = 0; level < node.links.size(); ++level) {
                        size_t begin, finish;
                        native.hnsw.neighbor_range(id, level, &begin, &finish);
                        for (size_t edge = begin; edge < finish; ++edge) {
                            const auto neighbor = native.hnsw.neighbors[edge];
                            if (neighbor == -1)
                                break;
                            if (neighbor < 0 || size_t(neighbor) >= total)
                                throw std::runtime_error(
                                        "invalid native fixture edge");
                            node.links[level].push_back(neighbor);
                        }
                    }
                    batch.nodes.emplace(
                            static_cast<ft::NodeId>(id), std::move(node));
                    batch.vectors.emplace(
                            static_cast<ft::NodeId>(id),
                            std::vector<float>(
                                    vectors.data() + id * o.dimensions,
                                    vectors.data() + (id + 1) * o.dimensions));
                }
                storage.apply(batch);
            }
            materialize_ms = milliseconds(start);
            const auto validate_start = Clock::now();
            const auto report = index.validate();
            fixture_validation_ms = milliseconds(validate_start);
            if (report.nodes != total || report.live_nodes != o.n)
                throw std::runtime_error("invalid materialized fixture counts");
        }
        commit_ms = tx.commit();
        max_writer_ms = milliseconds(writer_start);
        transactions = 1;
    }
    check(mdb_env_sync(db.env, 1));
    const auto file_bytes = std::filesystem::file_size(db.path + "/data.mdb");
    MDB_envinfo info{};
    MDB_stat stat{};
    check(mdb_env_info(db.env, &info));
    check(mdb_env_stat(db.env, &stat));
    std::vector<faiss::idx_t> allowed;
    for (size_t id = replaced; id < total; ++id) {
        const double uniform = static_cast<double>(hash(id ^ o.seed) >> 11) /
                9007199254740992.0;
        if (uniform < o.selectivity)
            allowed.push_back(id);
    }
    faiss::IDSelectorBatch selector(allowed.size(), allowed.data());
    faiss::SearchParametersHNSW params;
    params.efSearch = static_cast<int>(o.ef);
    // No selector on an unfiltered/no-churn run: compare the normal fast paths.
    params.sel = (replaced || o.selectivity < 1) ? &selector : nullptr;
    faiss::SearchParameters exact_params;
    exact_params.sel = params.sel;
    faiss::IndexFlat exact(o.dimensions, metric);
    exact.add(total, vectors.data());
    std::vector<std::unordered_set<faiss::idx_t>> truth(o.queries);
    std::vector<faiss::idx_t> labels(o.k);
    std::vector<float> distances(o.k);
    const auto exact_start = Clock::now();
    for (size_t q = 0; q < o.queries; ++q) {
        exact.search(
                1,
                queries.data() + q * o.dimensions,
                o.k,
                distances.data(),
                labels.data(),
                &exact_params);
        for (auto id : labels)
            if (id >= 0)
                truth[q].insert(id);
    }
    const double exact_ms = milliseconds(exact_start);
    ft::SearchOptions options;
    options.ef_search = o.ef;
    options.max_visits = o.max_visits;
    options.cache_bytes = multiply(o.cache_mib, 1u << 20);
    options.selector = params.sel;
    options.exact = o.mode == "exact";
    std::vector<ft::NodeId> seeds;
    if (o.mode == "acorn") {
        options.selector = &selector;
        options.acorn = true;
        auto candidates = allowed;
        const auto less = [&](auto a, auto b) {
            return hash(a ^ o.seed) < hash(b ^ o.seed);
        };
        const size_t count = std::min<size_t>(10, candidates.size());
        std::partial_sort(
                candidates.begin(),
                candidates.begin() + count,
                candidates.end(),
                less);
        for (size_t i = 0; i < count; ++i)
            seeds.push_back(candidates[i]);
        options.seed_ids = seeds;
    }
    ft::SearchStats work;
    size_t limited_queries = 0;
    Summary lmdb_summary, native_summary;
    Transaction read(db.env, true);
    ft::LmdbStorage storage(read.txn, db.table, 1);
    ft::HNSW index(storage);
    for (size_t i = 0; i < o.warmup; ++i) {
        const auto query = queries.data() + (i % o.queries) * o.dimensions;
        index.search(query, o.k, options);
        native.search(1, query, o.k, distances.data(), labels.data(), &params);
    }
    for (size_t repeat = 0; repeat < o.repeats; ++repeat) {
        for (size_t q = 0; q < o.queries; ++q) {
            const auto query = queries.data() + q * o.dimensions;
            const auto lmdb_query = [&] {
                const auto start = Clock::now();
                ft::SearchStats stats;
                auto found = index.search(query, o.k, options, &stats);
                const double ms = milliseconds(start);
                work.visited += stats.visited;
                work.expanded += stats.expanded;
                work.node_reads += stats.node_reads;
                work.vector_reads += stats.vector_reads;
                work.vector_block_reads += stats.vector_block_reads;
                work.live_block_reads += stats.live_block_reads;
                limited_queries += stats.visit_limit_reached;
                std::vector<faiss::idx_t> ids;
                ids.reserve(found.size());
                for (auto result : found)
                    ids.push_back(result.id);
                lmdb_summary.add(ids, truth[q], ms, total, selector);
            };
            const auto native_query = [&] {
                const auto start = Clock::now();
                native.search(
                        1,
                        query,
                        o.k,
                        distances.data(),
                        labels.data(),
                        &params);
                native_summary.add(
                        labels, truth[q], milliseconds(start), total, selector);
            };
            // Alternate execution order to reduce a systematic order advantage.
            if ((q + repeat) % 2) {
                native_query();
                lmdb_query();
            } else {
                lmdb_query();
                native_query();
            }
        }
    }
    std::cout << std::setprecision(9)
              << "{\n  \"workload\":{\"distribution\":\"normal\",\"n\":" << o.n
              << ",\"dimensions\":" << o.dimensions
              << ",\"queries\":" << o.queries << ",\"k\":" << o.k
              << ",\"batch\":" << o.batch << ",\"m\":" << o.m
              << ",\"ef_construction\":" << o.ef_construction
              << ",\"ef_search\":" << o.ef << ",\"max_visits\":" << o.max_visits
              << ",\"cache_mib\":" << o.cache_mib << ",\"lmdb_mode\":\""
              << o.mode << "\""
              << ",\"layout\":\"" << o.layout << "\""
              << ",\"requested_vectors_per_block\":" << o.vectors_per_block
              << ",\"build_mode\":\"" << o.build_mode << "\""
              << ",\"same_topology\":"
              << (o.build_mode == "native-materialized" ? "true" : "false")
              << ",\"seed\":" << o.seed << ",\"metric\":\"" << o.metric
              << "\",\"selectivity\":" << o.selectivity
              << ",\"allowed\":" << allowed.size() << ",\"churn\":" << o.churn
              << ",\"replaced\":" << replaced << ",\"warmup\":" << o.warmup
              << ",\"repeats\":" << o.repeats
              << ",\"omp_threads\":1,\"map_bytes\":" << map_size << "},\n"
              << "  \"build\":{\"lmdb_insert_and_erase_ms\":" << lmdb_insert_ms
              << ",\"lmdb_commit_ms\":" << commit_ms
              << ",\"native_insert_ms\":" << native_insert_ms
              << ",\"fixture_materialize_ms\":" << materialize_ms
              << ",\"fixture_validation_ms\":" << fixture_validation_ms
              << ",\"transactions\":" << transactions
              << ",\"max_writer_ms\":" << max_writer_ms
              << ",\"lmdb_file_bytes\":" << file_bytes
              << ",\"lmdb_high_water_pages\":" << info.me_last_pgno + 1
              << ",\"lmdb_page_bytes\":" << stat.ms_psize << "},\n"
              << "  \"exact_search_total_ms\":" << exact_ms << ",\n  \"lmdb\":";
    lmdb_summary.print();
    std::cout << ",\n  \"native\":";
    native_summary.print();
    std::cout << ",\n  \"lmdb_search_work\":{\"mean_distance_evaluations\":"
              << double(work.visited) / lmdb_summary.latency.size()
              << ",\"mean_expanded\":"
              << double(work.expanded) / lmdb_summary.latency.size()
              << ",\"mean_node_reads\":"
              << double(work.node_reads) / lmdb_summary.latency.size()
              << ",\"mean_vector_reads\":"
              << double(work.vector_reads) / lmdb_summary.latency.size()
              << ",\"mean_vector_block_reads\":"
              << double(work.vector_block_reads) / lmdb_summary.latency.size()
              << ",\"mean_live_block_reads\":"
              << double(work.live_block_reads) / lmdb_summary.latency.size()
              << ",\"budget_limited_queries\":" << limited_queries << "}\n}\n";
}
} // namespace
int main(int argc, char** argv) {
    try {
        run(parse(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}

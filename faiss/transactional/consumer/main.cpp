/* Copyright (c) Aperture Data Inc. Licensed under the root MIT LICENSE. */
#include <faiss/transactional/LmdbStorage.h>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    char pattern[] = "/tmp/faiss-consumer-XXXXXX";
    const char* directory = mkdtemp(pattern);
    if (!directory)
        return 1;
    MDB_env* env = nullptr;
    MDB_txn* txn = nullptr;
    int result = 1;
    try {
        const auto check = [](int status) {
            if (status)
                throw std::runtime_error(mdb_strerror(status));
        };
        check(mdb_env_create(&env));
        check(mdb_env_open(env, directory, 0, 0600));
        check(mdb_txn_begin(env, nullptr, 0, &txn));
        MDB_dbi table;
        check(mdb_dbi_open(txn, nullptr, 0, &table));
        {
            faiss::transactional::LmdbStorage storage(txn, table, 1);
            faiss::transactional::HNSW index(storage);
            faiss::transactional::Config config;
            config.dimensions = 2;
            index.create(config);
            const float vectors[] = {0, 0, 1, 1};
            index.add(2, vectors);
            const auto found = index.search(vectors + 2, 1);
            if (found.size() != 1 || found[0].id != 1) {
                throw std::runtime_error("incorrect search result");
            }
        }
        MDB_txn* completed = txn;
        txn = nullptr;
        check(mdb_txn_commit(completed));
        result = 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
    }
    if (txn)
        mdb_txn_abort(txn);
    if (env)
        mdb_env_close(env);
    std::filesystem::remove_all(directory);
    return result;
}

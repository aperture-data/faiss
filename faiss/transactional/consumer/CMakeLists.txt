# Build against the installed package, without discovering its dependencies first.
cmake_minimum_required(VERSION 3.24)
project(faiss_transactional_consumer LANGUAGES CXX)
find_package(faiss CONFIG REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE faiss_lmdb)
enable_testing()
add_test(NAME installed_package COMMAND consumer)

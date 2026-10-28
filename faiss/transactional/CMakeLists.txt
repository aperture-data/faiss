# Copyright (c) Aperture Data Inc. Licensed under the root MIT LICENSE.
add_library(faiss_transactional HNSW.cpp)
target_link_libraries(faiss_transactional PUBLIC faiss)
target_compile_features(faiss_transactional PUBLIC cxx_std_20)
set_target_properties(faiss_transactional PROPERTIES POSITION_INDEPENDENT_CODE ON)
install(TARGETS faiss_transactional EXPORT faiss-targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES HNSW.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/faiss/transactional)

if(FAISS_ENABLE_LMDB)
  include(${PROJECT_SOURCE_DIR}/cmake/FindFaissLMDB.cmake)
  add_library(faiss_lmdb LmdbStorage.cpp)
  target_link_libraries(faiss_lmdb PUBLIC faiss_transactional FAISS::LMDB)
  set_target_properties(faiss_lmdb PROPERTIES POSITION_INDEPENDENT_CODE ON)
  install(TARGETS faiss_lmdb EXPORT faiss-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(FILES LmdbStorage.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/faiss/transactional)
  install(FILES ${PROJECT_SOURCE_DIR}/cmake/FindFaissLMDB.cmake
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/faiss)
endif()

if(FAISS_TRANSACTIONAL_TESTING)
  if(NOT FAISS_ENABLE_LMDB)
    message(FATAL_ERROR "FAISS_TRANSACTIONAL_TESTING requires FAISS_ENABLE_LMDB")
  endif()
  find_package(GTest CONFIG REQUIRED)
  add_executable(faiss_transactional_test tests.cpp)
  target_link_libraries(faiss_transactional_test PRIVATE faiss_lmdb GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(faiss_transactional_test)
endif()

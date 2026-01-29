cmake_minimum_required(VERSION 3.0...3.26)
set(CMAKE_BUILD_TYPE Debug)
project(hnswlib
    LANGUAGES CXX C)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

# include_directories(hnswlib/cqf-master/include hnswlib/cqf-master/src)
# #file(GLOB CQF_SOURCES "external/cqf/src/.c" "external/cqf/include/.h")
add_compile_definitions(
    LANG=cpp
    PC_COUNTER
)

# Source files
set(SOURCES

    predictive_point_queries/count_min_sketch_min_hash.cpp

)


add_library(hnswlib STATIC ${SOURCES})
add_library(hnswlib::hnswlib ALIAS hnswlib)

target_include_directories(hnswlib
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)


# OpenMP support
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(hnswlib PUBLIC OpenMP::OpenMP_CXX)
endif()

# Install
install(TARGETS hnswlib
    EXPORT hnswlibTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hnswlib
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(EXPORT hnswlibTargets
    FILE hnswlibConfig.cmake
    NAMESPACE hnswlib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hnswlib
)

# Examples and tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(HNSWLIB_EXAMPLES "Build examples and tests." ON)
else()
    option(HNSWLIB_EXAMPLES "Build examples and tests." OFF)
endif()
if(HNSWLIB_EXAMPLES)
    set(CMAKE_CXX_STANDARD 11)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        SET(CMAKE_CXX_FLAGS "-Ofast -std=c++11 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize")
        check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
        if(COMPILER_SUPPORT_NATIVE_FLAG)
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
            message("set -march=native flag")
        else()
            check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
            if(COMPILER_SUPPORT_M1_FLAG)
                SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1")
                message("set -mcpu=apple-m1 flag")
            endif()
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        SET(CMAKE_CXX_FLAGS "-Ofast -lrt -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        SET(CMAKE_CXX_FLAGS "/O2 -DHAVE_CXX0X /W1 /openmp /EHsc")
    endif()

    # examples

    # Original examples
    add_executable(example_point_search examples/cpp/example_point_search.cpp)
    target_link_libraries(example_point_search hnswlib)

    add_executable(example_range_search examples/cpp/example_range_search.cpp)
    target_link_libraries(example_range_search hnswlib)

    add_executable(indexing_construction examples/cpp/index_construction.cpp)
    target_link_libraries(indexing_construction hnswlib)

    add_executable(example_conjunction_search examples/cpp/example_conjunction_search.cpp)
    target_link_libraries(example_conjunction_search hnswlib)
    # tests

endif()

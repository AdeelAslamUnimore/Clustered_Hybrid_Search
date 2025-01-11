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
    
    hnswlibcextensionfiles/count_min_sketch.c
    hnswlibcextensionfiles/bloom.c
    hnswlibcextensionfiles/partitioned_counter.c
    hnswlibcextensionfiles/hashutil.c
    hnswlibcextensionfiles/count_min_sketch_min_hash.cpp
    
)
# writing the compile instruction 
option(USE_CQF "Use cq method" OFF)
option(USE_ARRAY "Use array-based method" OFF)
option(USE_BLOOM_FILTER "Use Bloom Filter method" OFF)

# Conditionally add definitions based on options
if(USE_CQF)
    add_definitions(-DQF)
endif()
if(USE_ARRAY)
    add_definitions(-Darray_based)
endif()

if(USE_BLOOM_FILTER)
    add_definitions(-Dbloom_filter)
endif()
# SDSL Lite A Use Full Lib





add_library(hnswlib STATIC ${SOURCES})
add_library(hnswlib::hnswlib ALIAS hnswlib)

target_include_directories(hnswlib 
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:/home/u6059148/hnswlib/EigenLib/eigen> 
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

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      SET( CMAKE_CXX_FLAGS  "-Ofast -std=c++11 -DHAVE_CXX0X -openmp -fpic -ftree-vectorize" )
      check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
      if(COMPILER_SUPPORT_NATIVE_FLAG)
        SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native" )
        message("set -march=native flag")
      else()
        check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
        if(COMPILER_SUPPORT_M1_FLAG)
          SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1" )
          message("set -mcpu=apple-m1 flag")
        endif()
      endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      SET( CMAKE_CXX_FLAGS  "/O2 -DHAVE_CXX0X /W1 /openmp /EHsc" )
    endif()

    # examples
  

    add_executable(example_filter examples/cpp/example_filter.cpp)
    target_link_libraries(example_filter  hnswlib )
    add_executable(example_filter_index_creation examples/cpp/example_filter_index_creation.cpp)
    target_link_libraries(example_filter_index_creation  hnswlib)

    # tests
   
   endif()

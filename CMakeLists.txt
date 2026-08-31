cmake_minimum_required(VERSION 3.16)
project(qcsi VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(QCSI_BUILD_TESTS "Build unit tests" ON)
option(QCSI_WERROR      "Treat warnings as errors" ON)
option(QCSI_BENCH       "Build the cost and footprint report" ON)
option(QCSI_PROFILE     "Count fixed-point operations (bench only)" OFF)

include(FetchContent)

# qcsi is not a rewrite of the numeric core: it depends on qdsp for
# fixed-point arithmetic, filters and FFT. Pinning a tag rather than a branch
# keeps the build reproducible.
FetchContent_Declare(
    qdsp
    GIT_REPOSITORY https://github.com/andrealo20/qdsp.git
    GIT_TAG        0393071f32b56804723188567f4a858268f96144
)
set(QDSP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(QDSP_BENCH      OFF CACHE BOOL "" FORCE)
set(QDSP_EXAMPLES   OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(qdsp)

add_library(qcsi
    src/phase.c
    src/cordic_table.c
    src/features.c
    src/classify.c
    src/pipeline.c
)

target_include_directories(qcsi PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(qcsi PUBLIC qdsp)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(qcsi PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-prototypes)
    if(QCSI_WERROR)
        target_compile_options(qcsi PRIVATE -Werror)
    endif()
endif()

if(QCSI_PROFILE)
    target_compile_definitions(qdsp PUBLIC QDSP_PROFILE=1)
endif()

if(QCSI_BENCH)
    add_subdirectory(bench)
endif()

if(QCSI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

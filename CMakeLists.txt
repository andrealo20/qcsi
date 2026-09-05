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

# qdsp's FFT calls cosf and the project does not link libm itself, so every
# consumer has to supply it. Attaching it to the qdsp target is what puts it
# in the right place: CMake emits a target's interface libraries immediately
# after the target, so libm always follows libqdsp on the link line. Declaring
# it on qcsi instead left the two as siblings, in an order CMake was free to
# choose, and it chose wrong for exactly one executable.
include(CheckLibraryExists)
check_library_exists(m cos "" QCSI_HAVE_LIBM)
if(QCSI_HAVE_LIBM AND TARGET qdsp)
    target_link_libraries(qdsp PUBLIC m)
endif()

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

# qdsp's FFT uses cosf, so libm has to appear after libqdsp on the link line.
# Declaring it here rather than on each executable is what keeps the order
# right: every consumer inherits it through qcsi, after qdsp, because that is
# the order these two are declared in. Attaching it per test target instead
# left it to chance, and test_parity was the one where the chance came up
# wrong.
include(CheckLibraryExists)
check_library_exists(m cos "" QCSI_HAVE_LIBM)
if(QCSI_HAVE_LIBM)
    target_link_libraries(qcsi PUBLIC m)
endif()

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

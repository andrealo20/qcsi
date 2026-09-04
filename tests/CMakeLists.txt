include(FetchContent)

FetchContent_Declare(
    unity
    GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git
    GIT_TAG        v2.6.0
)
FetchContent_MakeAvailable(unity)

target_compile_definitions(unity PUBLIC UNITY_INCLUDE_DOUBLE)

set(QCSI_TEST_MODULES phase features classify pipeline parity)

foreach(mod IN LISTS QCSI_TEST_MODULES)
    add_executable(test_${mod} test_${mod}.c)
    target_link_libraries(test_${mod} PRIVATE qcsi unity)
    target_include_directories(test_${mod} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_${mod} PRIVATE -Wall -Wextra)
        target_link_libraries(test_${mod} PRIVATE m)
    endif()
    add_test(NAME ${mod} COMMAND test_${mod})
endforeach()

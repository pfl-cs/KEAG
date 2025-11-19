include_directories(../include)

function(target_source_tree target)
    #    file(GLOB_RECURSE SRCS ${ARGN})
    file(GLOB SRCS ${ARGN})
    target_sources(${target} PRIVATE ${SRCS})
endfunction()


set(DIR rtree)
set(SOURCES
        AccessGT
        Eval)

set(CONFIG_DIR config)
set(DATA_PROCESS_DIR data_process)

add_library(config SHARED config)
target_source_tree(config
        ${CONFIG_DIR}/*.cc)
message(STATUS "config is added")

add_library(data_process SHARED data_process)
target_source_tree(data_process
        ${DATA_PROCESS_DIR}/*.cc)
message(STATUS "data_process is added")


include_directories(config)
include_directories(data_process)
target_link_libraries(config PRIVATE spatialindex_utils)
target_link_libraries(data_process PRIVATE config)
target_link_libraries(data_process PRIVATE spatialindex_utils)
target_link_libraries(data_process PRIVATE spatialindex)


foreach(test ${SOURCES})
    add_executable(${test} ${DIR}/${test}.cc)
    target_link_libraries(${test} PRIVATE spatialindex_utils)
    target_link_libraries(${test} PRIVATE spatialindex)
    target_link_libraries(${test} PRIVATE config)
    target_link_libraries(${test} PRIVATE data_process)
endforeach()


if (UNIX)
    add_test(NAME index-tests
             COMMAND /bin/sh "${CMAKE_CURRENT_SOURCE_DIR}/index-tests.sh"
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_property(TEST index-tests PROPERTY ENVIRONMENT
             "PATH=${CMAKE_BINARY_DIR}/test/:$ENV{PATH}")
endif (UNIX)

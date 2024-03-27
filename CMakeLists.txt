
cmake_minimum_required(VERSION 3.0)


set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

project(kerasCpp)

set(SOURCES fileio.cc)

set(HEADERS fileio.h)


add_library(libfileio STATIC ${SOURCES} ${HEADERS})

set(SOURCES layers.cc)
set(HEADERS layers.h)

add_library(liblayers STATIC ${SOURCES} ${HEADERS})

add_executable(example_main example_main.cc)

target_link_libraries(example_main PUBLIC libfileio liblayers)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/example)
file(COPY example/sample_mnist.dat DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/example)
file(COPY example/dumped_mnist.nnet DESTINATION ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/example)


cmake_minimum_required(VERSION 3.10)
project(PythiaTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")

find_package(Torch REQUIRED)

add_executable(test_nn
    test/test_learning_engine_nn.cc
    src/learning_engine_neural_network.cc
)

target_include_directories(test_nn PRIVATE
    src
    inc
)

target_link_libraries(test_nn "${TORCH_LIBRARIES}")
set_property(TARGET test_nn PROPERTY CXX_STANDARD 17)

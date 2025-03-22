# This is used to build the test for the learning_engine_neural_network
cmake_minimum_required(VERSION 3.10)  # Minimum required CMake version
project(PythiaTest) # Define the project name

set(CMAKE_CXX_STANDARD 17)  # Use C++17 standard
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")  # Path to LibTorch

find_package(Torch REQUIRED)  # Find the Torch package

# Define the test executable and its source files
add_executable(test_nn
    test/test_learning_engine_nn.cc
    src/learning_engine_neural_network.cc
)

# Set the include directories
target_include_directories(test_nn PRIVATE
    src
    inc
)
# Link the Torch libraries
target_link_libraries(test_nn "${TORCH_LIBRARIES}")

# Ensure C++17 is used for this target
set_property(TARGET test_nn PROPERTY CXX_STANDARD 17)

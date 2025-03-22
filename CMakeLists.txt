
cmake_minimum_required(VERSION 3.10)
project(ScoobyWithNN)

set(CMAKE_CXX_STANDARD 14)

# Path to LibTorch
set(Torch_DIR "/usr/local/lib/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# Source files
set(SOURCE_FILES scooby.cc)

# Create executable
add_executable(ScoobyWithNN ${SOURCE_FILES})
target_link_libraries(ScoobyWithNN "${TORCH_LIBRARIES}")
set_property(TARGET ScoobyWithNN PROPERTY CXX_STANDARD 14)

# Set flags to enable all optimizations and debugging
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g")

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# 1. Set compiler paths BEFORE the project() call
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.0/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4" CACHE PATH "Path to OpenCV" FORCE)

project(Cuda_app LANGUAGES CXX CUDA)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

# 2. Find OpenCV
find_package(OpenCV REQUIRED)

# 3. Define executable
add_executable(Cuda_app FFT_Algorithms.cu)

# 4. Standards
set_target_properties(Cuda_app PROPERTIES 
    CXX_STANDARD 17
    CUDA_STANDARD 17
)

# 5. Include and LINK (Crucial step)
target_include_directories(Cuda_app PUBLIC ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Cuda_app PUBLIC ${OpenCV_LIBRARIES})
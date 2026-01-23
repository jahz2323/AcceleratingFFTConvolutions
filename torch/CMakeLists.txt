
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
# Add this to the TOP of your CMakeLists.txt (after cmake_minimum_required)
set(Torch_DIR "/home/jahz/libtorch/share/cmake/Torch" CACHE PATH "Path to Torch" FORCE)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.0/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4" CACHE PATH "Path to OpenCV" FORCE)
set(CMAKE_CUDA_ARCHITECTURES 89)

# setup debug
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(torchapp LANGUAGES CXX CUDA)

# 3. Find the LibTorch package
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(customops)
add_subdirectory(models)
add_subdirectory(utils)

# target_link_libraries(torchapp_ops PRIVATE "${TORCH_LIBRARIES}")
# target_compile_features(torchapp_ops PRIVATE cxx_std_17)

# 4. Define your executable (include both .cpp and .cu files here)
add_executable(torchapp main.cpp)

# 5. Link the LibTorch libraries
target_link_libraries(torchapp PRIVATE utils)

# 6. Set C++ and CUDA standards
target_compile_features(torchapp PRIVATE cxx_std_17)
set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
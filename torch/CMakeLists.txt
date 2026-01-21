cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Add this to the TOP of your CMakeLists.txt (after cmake_minimum_required)
set(Torch_DIR "/home/jahz/libtorch/share/cmake/Torch" CACHE PATH "Path to Torch" FORCE)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.0/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4" CACHE PATH "Path to OpenCV" FORCE)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(SOURCES 
    ${CMAKE_SOURCE_DIR}/models/AlexNet.cpp
    ${CMAKE_SOURCE_DIR}/models/AlexNet.hpp
)
# setup debug
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(torchapp LANGUAGES CXX CUDA)

# 3. Find the LibTorch package
find_package(Torch REQUIRED)
# 3.1 Find OpenCV
find_package(OpenCV REQUIRED)

# Custom operator library
add_library(torchapp_ops SHARED 
    customops/operators/extension.cpp
    customops/kernels/all_reduce.cu
)

target_link_libraries(torchapp_ops PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(torchapp_ops PUBLIC 
    ${CMAKE_SOURCE_DIR}/customops/operators
    ${CMAKE_SOURCE_DIR}/customops/kernels
)
target_compile_features(torchapp_ops PRIVATE cxx_std_17)

# 4. Define your executable (include both .cpp and .cu files here)
add_executable(torchapp dataprep.cpp main.cpp ${SOURCES})


target_compile_features(torchapp PRIVATE cxx_std_17)
# 4.1 INCLUDE DIRS 
target_include_directories(torchapp PUBLIC 
    ${OpenCV_INCLUDE_DIRS} 
    ${CMAKE_SOURCE_DIR}/customops/operators
    ${CMAKE_SOURCE_DIR}/customops/kernels
)

# 5. Link the LibTorch libraries
target_link_libraries(torchapp "${TORCH_LIBRARIES}" ${OpenCV_LIBS} torchapp_ops)

# 6. Set C++ and CUDA standards
set_property(TARGET torchapp_ops PROPERTY CXX_STANDARD 17)
set_property(TARGET torchapp_ops PROPERTY CUDA_STANDARD 17)
set_property(TARGET torchapp PROPERTY CXX_STANDARD 17)
set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
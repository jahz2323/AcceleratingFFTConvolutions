cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# 1. Set compiler paths BEFORE the project() call
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.0/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
set(Torch_DIR "/home/jahz/libtorch/share/cmake/Torch" CACHE PATH "Path to Torch" FORCE)
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4" CACHE PATH "Path to OpenCV" FORCE)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 89)

project(Cuda_app LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Development NumPy REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)
# find_package(fmt REQUIRED)

add_subdirectory(tests)
add_subdirectory(operations)
add_subdirectory(libs)



# 3. Define executable
add_executable(Cuda_app 
    main.cpp

)

# 4. Standards
set_target_properties(Cuda_app PROPERTIES 
    CXX_STANDARD 23
    CUDA_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CUDA_STANDARD_REQUIRED YES
)
target_link_libraries(Cuda_app PRIVATE -Wl,--start-group
    tests 
    operations 
    libs
    -Wl,--end-group
    "${TORCH_LIBRARIES}" 
    ${OpenCV_LIBS}
    CURL::libcurl 
    # fmt::fmt
)

#5. Create directories for results, if found clear contents of results directory
set(results_dir "${CMAKE_BINARY_DIR}/results")
set(images_dir "${results_dir}/images")
set(mse_dir "${results_dir}/mse")
file(MAKE_DIRECTORY ${images_dir})
file(MAKE_DIRECTORY ${mse_dir})
file(GLOB images_files "${images_dir}/*")
file(GLOB mse_files "${mse_dir}/*")
foreach(file ${images_files})
    file(REMOVE ${file})
endforeach()
foreach(file ${mse_files})
    file(REMOVE ${file})
endforeach()    

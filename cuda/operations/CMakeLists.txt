add_library(operations STATIC src/cuda_operations.cu src/utils.cu)
target_link_libraries(operations PUBLIC CUDA::cufft libs ${OpenCV_LIBS} )
target_include_directories(operations PUBLIC include
    cuda_std_20
    ${OpenCV_INCLUDE_DIRS}
)
add_library(operations STATIC src/cuda_operations.cu src/utils.cu)
target_link_libraries(operations PUBLIC CUDA::cufft libs)
target_include_directories(operations PUBLIC include)
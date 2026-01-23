add_library(customops SHARED kernels/src/all_reduce.cu operators/src/extension.cpp)

target_link_libraries(customops PUBLIC "${TORCH_LIBRARIES}")

target_include_directories(customops PUBLIC
    include 
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/include
)

set_target_properties(
    customops
    PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
)
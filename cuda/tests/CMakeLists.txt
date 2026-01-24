add_library(tests STATIC src/convolution_test.cu)
target_link_libraries(tests PUBLIC operations)
target_include_directories(tests PUBLIC 
    include
    cuda_std_20
)
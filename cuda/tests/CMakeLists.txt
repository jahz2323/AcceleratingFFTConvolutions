add_library(tests STATIC src/convolution_test.cu)
target_link_libraries(tests PUBLIC 
    operations
    ${OpenCV_LIBS} 
)
target_include_directories(tests PUBLIC 
    include
    cuda_std_20
    ${OpenCV_INCLUDE_DIRS}
)
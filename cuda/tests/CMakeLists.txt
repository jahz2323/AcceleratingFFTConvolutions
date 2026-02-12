add_library(tests STATIC src/convolution_test.cu)

target_sources(tests PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/../../torch/utils/src/dataprep.cpp"
)

target_include_directories(tests PUBLIC 
    include
    cuda_std_20
    ${OpenCV_INCLUDE_DIRS}
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../torch/utils/include" 
)

target_link_libraries(tests PUBLIC 
    operations
    ${OpenCV_LIBS} 
    ${TORCH_LIBRARIES}
)

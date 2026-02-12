add_library(libs INTERFACE)
target_link_libraries(libs INTERFACE torch Python3::Python Python3::NumPy ${OpenCV_LIBS})
target_include_directories(libs INTERFACE 
    include
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)
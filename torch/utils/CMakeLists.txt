add_library(utils STATIC 
    src/dataprep.cpp 
    src/inference.cpp 
    src/train.cpp
)
target_link_libraries(utils PUBLIC models 
    ${OpenCV_LIBS} 
    ${TORCH_LIBRARIES} 
)

target_include_directories(utils PUBLIC 
    include 
    ${CMAKE_CURRENT_SOURCE_DIR}/models/include 
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)
add_library(models SHARED src/AlexNet.cpp src/resnet.cpp)

target_link_libraries(models PUBLIC customops torch)

target_include_directories(models PUBLIC include)

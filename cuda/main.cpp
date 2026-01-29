#include <iostream>
#include <string>
#include "convolution_test.cuh"

enum class TestMode {
    CONVOLUTION,
    SFFT,
    CUSTOM_OPS,
    GPUSFFT
};

/**
    @brief Entry for CUDA operation testing 
    1. Run Convolution tests
    2. GPU-SFFT convolution tests 
    3. Custom Operations test 
*/
int main(int argc, char* argv[]){
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mode> [additional arguments]" << std::endl;
        std::cerr << "Modes:" << std::endl;
        std::cerr << "test_conv" << std::endl;
        std::cerr << "test_sfft" << std::endl;
        std::cerr << "test_custom_ops" << std::endl;
        std::cerr << "test_gpusfft" << std::endl;
        return 1;
    }
    std::string mode = argv[1];
    if (mode == "test_conv") {
        // Call convolution test function
        std::cout << "Running convolution tests..." << std::endl;
        convolution_test::convolve();
    } else if (mode == "test_sfft") {
        // Call SFFT test function
        std::cout << "Running SFFT tests..." << std::endl;
        // sfft::runTests();
    } else if (mode == "test_custom_ops") {
        // Call custom operations test function
        std::cout << "Running custom operations tests..." << std::endl;
        // custom_ops::runTests();
    } else if (mode == "test_gpusfft") {
        // Call GPU-SFFT test function
        std::cout << "Running GPU-SFFT tests..." << std::endl;
        // gpusfft::runTests();
    } 
    else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }
}
#include <iostream>
#include <string>
#include "convolution_test.cuh"

#include <limits> // check IEEE 754 Complicant 

void ieee_standard_check() {
    if (std::numeric_limits<float>::is_iec559) {
        std::cout << "Float is IEEE 754 compliant." << std::endl;
    }
    if (std::numeric_limits<double>::is_iec559) {
        std::cout << "Double is IEEE 754 compliant." << std::endl;
    }
}
/**
    @brief Entry for CUDA operation testing 
    1. Run Convolution tests
    2. GPU-SFFT convolution tests 
    3. Custom Operations test 
*/
int main(int argc, char* argv[]){
    ieee_standard_check();
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mode> [additional arguments]" << std::endl;
        std::cerr << "Modes:" << std::endl;
        std::cerr << "test_conv" << std::endl;
        std::cerr << "test_custom_ops" << std::endl;
        std::cerr << "benchmark" << std::endl;
    
        return 1;
    }
    std::string mode = argv[1];
    if (mode == "test_conv") {
        // Call convolution test function
        std::cout << "Running convolution tests..." << std::endl;
        std::string test_type = argv[2];
        if(test_type != "image" && test_type != "random") {
            std::cerr << "Please specify test type:" << std::endl;
            std::cerr << "image: for image convolution test" << std::endl;
            std::cerr << "random: for random data convolution test" << std::endl;
            return 1;
        }
        convolution_test::convolve(argv);
    }
    if (mode == "test_pool") {
        // Call pooling test function
        std::cout << "Running pooling tests..." << std::endl;
        convolution_test::pooling(argv);
    }
    else if (mode == "test_custom_ops") {
        // Call custom operations test function
        std::cout << "Running custom operations tests..." << std::endl;
        // custom_ops::runTests();
    } else if (mode == "benchmark") {
        std::cout << "Running benchmarks..." << std::endl;
        convolution_test::run_benchmarks();
    }
    else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }
}
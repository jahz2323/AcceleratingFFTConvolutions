#pragma once 

#include <chrono>
#include <iostream>
#include <torch/torch.h>
#include <math_constants.h>
#include <utils.cuh>
#include <cuda_operations.cuh>
using namespace cuda_operations;

// Define test modes
enum class TestMode {
    Random,
    Image,
    Unknown
};

inline TestMode parseMode(char* arg) {
    if (!arg) return TestMode::Unknown;
    std::string s(arg);
    if (s == "0" || s == "random") return TestMode::Random;
    if (s == "1" || s == "image")  return TestMode::Image;
    return TestMode::Unknown;
}


class convolution_test {
public:
    static void test1DConvolution();
    template <bool image_test> 
    static void test2DConvolution(int, int, int, int, int, int, cv::Mat, cv::Mat);
    static void testFFTConvolution();
    static void convolve(char* argv[]);
};
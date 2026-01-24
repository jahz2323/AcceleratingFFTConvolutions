#pragma once 
#include <cuda_operations.cuh>
#include <chrono>
#include <iostream>
#include <torch/torch.h>
#include <math_constants.h>
using namespace cuda_operations;

class convolution_test {
public:
    static void test1DConvolution();
    static void test2DConvolution();
    static void testFFTConvolution();
    static void convolve();
};
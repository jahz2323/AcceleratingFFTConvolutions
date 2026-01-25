#pragma once 
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuComplex.h>
#include <chrono>
#include <math_constants.h>
#include <string>
namespace utils{

    std::chrono::high_resolution_clock::time_point startTimer();
    float stopTimer(std::chrono::high_resolution_clock::time_point start);
    void MeasurementsToCSV(const std::string& filename); 


    __global__  void bitreversal(int width, int height, cuComplex* data);
    __global__ void bitreversal(int n, void* storage);
    __global__ void float2complex(int width, int height, float* input, cuComplex* output);
    __global__ void complex2float(int width, int height, cuComplex* input, float* output);
    __global__ void copy(float* odata, float* idata);
    __global__ void copyComplex(cuComplex* odata, cuComplex* idata);
    int nextPowerOfTwo(int n);
    __global__ void naivetranspose(int width, int height, cuComplex* input, cuComplex* output);
    
}
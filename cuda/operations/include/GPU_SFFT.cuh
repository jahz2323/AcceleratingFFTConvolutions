#pragma once 
#include <cuda_runtime.h>
#include <cufft.h>
#include <curand.h>
#include <iostream>
#include <vector>
#include <complex>
#include <filesystem>
#include "opencv.hpp"

class sfft{
    private: 
    cuComplex* d_input;
    float* d_output;
    int width;
    int height;
    
    __global__ void PFKernel(
        cuComplex* dbins, 
        cuComplex* dx, 
        cuComplex* dfilter, 
        int n,
        int B, 
        cuComplex* d_hsigma, 
        int i,
        int T, 
        int R 
    )

    /**
        @brief GPU Outer Loop for SFFT
        Input: hx[0..n],df iltt[0..f s],df iltf [0..f s]
        Output: ˆhx[0..IF ]
    */
    cuComplex* GPUOuterLoop(
        cuComplex* hx, // input in frequency domain
        int n, // input size
        cuComplex* dfilter_Time, // filter in time domain
        cuComplex* dilter_Freq, // filter in frequency domain 
        int fs, // filter size 
        int B, // number of bins (small fft size) - in location loops 
        int BT, // Estimation Bins - used in estimation loops , usually larger than B (2K , 4K)
        int W, // window width {window/size of filter being applied, determines how sharp the bins are}
        int L, // number of rounds for estimation {Number of hash rounds to find indices/coord of large freq}
        int Lc, // number of rounds in location loops {Number of hash rounds to find values of large freq}
        int Lt, // in location loop , index is voted for Max(LT,vote) to be considered significant
        int Ll, // Round limit - 
    )

    public:
    sfft(int, int);
    ~sfft();
    
}
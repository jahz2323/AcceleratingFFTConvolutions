#pragma once 
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuComplex.h>
#include <chrono>
#include <math_constants.h>
#include <string>
#include <fstream> 
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>




namespace utils{
    float MeasureError(const std::vector<float>& output1, const std::vector<float>& output2);
    int nextPowerOfTwo(int n);
    void checkcuComplexArray(cuComplex* data, int width, int height, const std::string& array_name);
    void writeCSV(
        const std::string& path_to_file_with_filename, 
        const std::vector<std::string> &content,
        const std::vector<std::string> &headers
    );
    void saveOutputImage(
        const std::string& path_to_file_with_filename,
        const std::vector<float>& output,
        int out_width,
        int out_height
    );
    void printConvResult(std::vector<float>& output, int out_width, int out_height);
}
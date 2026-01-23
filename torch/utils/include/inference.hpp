#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <opencv2/opencv.hpp> 
#include <ATen/cuda/CUDAContext.h>
#include <time.h>
#include <thread> // might parallelize data loading
#include <chrono>
#include <dlfcn.h>
#include "dataprep.hpp" 
#include "AlexNet.hpp"


namespace inference {
    /**
        @brief Run inference on trained alexnet-cifar10 model. 
        Get Top-1 and Top-5 accuracy on test dataset.
        Measure recall, precision, F1-score.
        @note: Img at path needs to be 32x32 RGB image 
        @param model_path: path to trained model file
        @param data_path: path to inference image file
        @param full_label_file_path: path to cifar10 label file (text file with labels)
    */
    void runInference(const std::string& model_path, const std::string& input_data, const std::string& full_label_file_path); 
} // namespace inference

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
#include "resnet.hpp"

#include "customAutograd.cuh"

namespace train{
    HandKeypoint KeypointFunction(const std::string &DATAPATH);
    void TestCustomOperator();
    void trainCIFAR();
    void trainResNet();
    CIFAR TestReadingCIFARBin(std::string root, CIFAR::Mode mode);
}

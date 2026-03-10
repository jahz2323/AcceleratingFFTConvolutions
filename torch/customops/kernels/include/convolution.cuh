#pragma once 
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>


torch::Tensor _2DConvLauncher(torch::Tensor input, torch::Tensor weights, int8_t stride, int8_t padding);
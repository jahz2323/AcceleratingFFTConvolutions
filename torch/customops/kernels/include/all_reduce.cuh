#pragma once 
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor all_reduce_launcher(torch::Tensor input);

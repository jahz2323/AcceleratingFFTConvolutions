#include <torch/torch.h>
#include "all_reduce.cuh"
#include "convolution.cuh"
static torch::Tensor custom_allreduce(torch::Tensor input) {
    return all_reduce_launcher(input);
}
static torch::Tensor custom_2DConv(torch::Tensor input, torch::Tensor filters, int8_t stride, int8_t padding) {
    return _2DConvLauncher(input, filters, stride, padding);
}

TORCH_LIBRARY (my_ops, m){
    m.def("custom_allreduce", &custom_allreduce);
    m.def("custom_2DConv", &custom_2DConv);
}
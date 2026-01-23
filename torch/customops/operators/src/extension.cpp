#include <torch/torch.h>
#include "all_reduce.cuh"

static torch::Tensor custom_allreduce(torch::Tensor input) {
    return all_reduce_launcher(input);
}

TORCH_LIBRARY (my_ops, m){
    m.def("custom_allreduce", &custom_allreduce);
}
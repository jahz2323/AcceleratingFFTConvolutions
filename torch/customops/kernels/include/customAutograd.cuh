#pragma once 
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include "customAutograd.cuh"

class myConv2DFunction : public torch::autograd::Function<myConv2DFunction>{
    public:
    inline static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor input, torch::Tensor filters, int8_t stride, int8_t padding){
        
        auto op = torch::Dispatcher::singleton()
                .findSchemaOrThrow("my_ops::custom_2DConv", "")
                .typed<torch::Tensor (torch::Tensor, torch::Tensor, int8_t, int8_t)>();


        auto output = op.call(input, filters, stride, padding);
        ctx->save_for_backward({input, filters, output, torch::tensor(stride), torch::tensor(padding)});
        return output;
    }

    inline static std::vector<torch::Tensor> backward(torch::autograd::AutogradContext *ctx, std::vector<torch::Tensor> grad_outputs){
        auto grad_output = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto filters = saved[1];
        static_cast<void>(saved[2]); // output, not needed for backward
        int8_t stride = saved[3].item<int8_t>();
        int8_t padding = saved[4].item<int8_t>();
        auto op = torch::Dispatcher::singleton()
                .findSchemaOrThrow("my_ops::custom_2DConv", "")
                .typed<torch::Tensor (torch::Tensor, torch::Tensor, int8_t, int8_t)>();
        // compute gradients for input and filters
        //create a update tensor for filters and input with the same shape as filters and input respectively, to store the gradients
 
        //1. Gradients wrt weights  DL/DW

        //DYpred/DW = X, DL/DYpred = grad_output, so DL/DW = X^T * grad_output
        auto grad_filters = input.sum({0,2,3}, /*keepdim */ true) * grad_output;

        //2. Gradients wrt input DL/DX
        //DYpred/DX = W, DL/DYpred = grad_output, so DL/DX = grad_output * W^T
        auto grad_input = filters.sum({0,2,3}, /*keepdim */ true) * grad_output;


        // grad_input = grad_input.expand_as(input);
        // grad_filters = grad_filters.expand_as(filters);
        return {grad_input, grad_filters, torch::Tensor()};
    }
};

class MyLinearFunction : public torch::autograd::Function<MyLinearFunction>{
    public:
    inline static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor input_tensor, torch::Tensor weights, torch::Tensor bias){
        // ctx is a context obj to save info for backward pass, 
        // input_tensor,weights bias are the input tensors for forward pass
        
        // perform op 
        auto output = torch::mm(input_tensor, weights.t()); 
        if (bias.defined()) {
            output += bias.unsqueeze(0).expand_as(output);
        }
        // save for backward 
        ctx->save_for_backward({input_tensor, weights, bias});
        return output;
    }

    inline static std::vector<torch::Tensor> backward(torch::autograd::AutogradContext *ctx, std::vector<torch::Tensor> grad_output){
        // ctx is the context obj to retrieve info from forward pass
        // grad_output is the gradient of the output from forward pass

        // retrieve saved tensors
        auto saved = ctx->get_saved_variables();
        auto input_tensor = saved[0];
        auto weights = saved[1];
        // auto bias = saved[2];

        // compute gradients for input_tensor, weights and bias
        auto grad_input = torch::mm(grad_output[0], weights);
        auto grad_weights = torch::mm(grad_output[0].t(), input_tensor);
        auto grad_bias = grad_output[0].sum(0);

        return {grad_input, grad_weights, grad_bias};
    }
};
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
        // auto op = torch::Dispatcher::singleton()
        //         .findSchemaOrThrow("my_ops::custom_2DConv", "")
        //         .typed<torch::Tensor (torch::Tensor, torch::Tensor, int8_t, int8_t)>();
        // compute gradients for input and filters
        //create a update tensor for filters and input with the same shape as filters and input respectively, to store the gradients
        // Conceptual: grad_weights = conv2d(input, grad_output)
        // You essentially treat the input channels as the 'batch' 
        // to get the gradients for each filter.

        //1. Gradients wrt weights  DL/DW

        //DYpred/DW = X, DL/DYpred = grad_output, so DL/DW = X^T * grad_output
        // 1. Weight Grad: Input convolved with Grad_Output
        // 2. Input Grad: Grad_Output convolved with flipped filters (and appropriate padding) WITH STRIDE taken into account
        // http://deeplearning.cs.cmu.edu/S21/document/slides/Lec12.CNN4.pdf
        /**
            Plan:
            Iterate through output channels
            Interate through input channels
            iterate through output grid 
            iterate through filter grid
        */
        auto grad_input = torch::zeros_like(input); 
        auto grad_filters = torch::zeros_like(filters);
        int64_t batch_size = input.size(0);

        //call op 
        auto op = torch::Dispatcher::singleton()
                .findSchemaOrThrow("my_ops::custom_2DConvBackward", "")
                .typed<std::tuple<torch::Tensor, torch::Tensor>(torch::Tensor, torch::Tensor, torch::Tensor, int8_t, int8_t)>();
        auto [grad_input_op, grad_filters_op] = op.call(grad_output, input, filters, stride, padding);
        // for (int64_t batch =0; batch < batch_size; ++batch){
        //     for (int j =0; j < filters.size(0); ++j){ // loop over output channels
        //         for (int i =0; i < filters.size(1); ++i){ // loop over input channels

        //             //Loop over output Grid 
        //             for ( int x = 0; x < grad_output.size(2); ++x){
        //                 for (int y = 0; y < grad_output.size(3); ++y){

        //                     // top left corner of the filter on the input
        //                     int x_start = x * stride - padding;
        //                     int y_start = y * stride - padding;

        //                     //upstream gradient 
        //                     float dz = grad_output[batch][j][x][y].item<float>();

        //                     //Loop over kernel 
        //                     for (int kx= 0; kx < filters.size(2); ++kx){
        //                         for (int ky = 0; ky < filters.size(3); ++ky){
        //                             int x_in = x_start + kx;
        //                             int y_in = y_start + ky;

        //                             // Check for valid input coordinates (handle padding)
        //                             if (x_in >= 0 && x_in < input.size(2) && y_in >= 0 && y_in < input.size(3)){
        //                                 // Update gradients
        //                                 grad_filters[j][i][kx][ky] += input[batch][i][x_in][y_in].item<float>() * dz; // DL/DW += X * DL/DYpred
        //                                 grad_input[batch][i][x_in][y_in] += filters[j][i][kx][ky].item<float>() * dz; // DL/DX += W * DL/DYpred
        //                             }
        //                         }
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }

        // // grad_input = grad_input.expand_as(input);
        // // grad_filters = grad_filters.expand_as(filters);
        return {grad_input, grad_filters, torch::Tensor(), torch::Tensor()};
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
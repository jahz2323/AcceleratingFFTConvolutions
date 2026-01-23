#pragma once
#include "torch/torch.h"
#include <opencv2/opencv.hpp>

cv::Mat torchTensorToOpenCVImg(torch::Tensor data){
    //1. prepare tensor: Move to cpu, remove batch dim and convert to 0-255
    torch::Tensor tensor_to_view = data.detach().cpu().squeeze();
    tensor_to_view = tensor_to_view.mul(255).to(torch::kUInt8); 

    //2. Create OpenCV Mat
    cv::Mat img(tensor_to_view.size(0), tensor_to_view.size(1), CV_8UC1, tensor_to_view.data_ptr());
    return img; 
}

struct Net : torch::nn::Module {
    Net() { 
    // C1: 1 input channel, 6 output channels, 5x5 kernel
    C1 = register_module("C1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 6, 5)));
    
    // S2: 2x2 average pooling with stride 2
    S2 = register_module("S2", torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2).stride(2)));
    
    // C3: 6 input channels (from S2), 16 output channels, 5x5 kernel
    C3 = register_module("C3", torch::nn::Conv2d(torch::nn::Conv2dOptions(6, 16, 5)));
    
    // S4: 2x2 average pooling with stride 2
    S4 = register_module("S4", torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2).stride(2)));
    
    // FC5: Input is 16 channels * 4 * 4 spatial dim = 256
    FC5 = register_module("FC5", torch::nn::Linear(16 * 4 * 4, 120));
    
    // FC6: 120 -> 84
    FC6 = register_module("FC6", torch::nn::Linear(120, 84));
    
    // FC7: 84 -> 47 (EMNIST Balanced has 47 classes. Use 10 if only using digits)
    FC7 = register_module("FC7", torch::nn::Linear(84, 47));
}

    // Implement the forward function
    torch::Tensor forward(torch::Tensor x) { // Removed & to allow chaining
        x = torch::relu(C1->forward(x));
        x = S2->forward(x); 
        x = torch::relu(C3->forward(x)); 
        x = S4->forward(x); 
        
        // Flatten: [Batch, 16, 4, 4] -> [Batch, 256]
        x = x.view({x.size(0), -1}); 
        
        x = torch::relu(FC5->forward(x)); 
        x = torch::relu(FC6->forward(x)); 
        return torch::log_softmax(FC7->forward(x), /*dim=*/ 1); 
    }
    // Layers
    torch::nn::Conv2d C1{nullptr}, C3{nullptr};
    torch::nn::AvgPool2d S2{nullptr}, S4{nullptr};
    torch::nn::Linear FC5{nullptr}, FC6{nullptr}, FC7{nullptr};
};

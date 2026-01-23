#pragma once
#include "torch/torch.h"
class AlexNet : public torch::nn::Module {
    private: 
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr}, conv5{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, softmax{nullptr};
    torch::nn::MaxPool2d maxpool1{nullptr}, maxpool2{nullptr}, maxpool3{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::Softmax softmax_layer{nullptr};
    public: 
    AlexNet();
    torch::Tensor forward(torch::Tensor x);
};


#pragma once
#include "torch/torch.h"

/**
    @Author Jahziel Angelo Belmonte
    Building ResNet Model from Deep Residual Learning for Image Recognition
    @ref author He et al., 2015
*/

/**
    Residual Block Structure
    Convolutional Layer -> BatchNorm -> ReLU -> Convolutional Layer -> addition -> ReLU 
*/
struct Residual_Block : public torch::nn::Module {
    private:
    // main path
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}; 
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::ReLU relu1{nullptr}, relu2{nullptr};

    // shortcut path
    torch::nn::Conv2d shortcut{nullptr};
    torch::nn::BatchNorm2d shortcut_bn{nullptr};
    public:
    Residual_Block(int in_channels, int out_channels, int stride, int filter_size);
    torch::Tensor forward(torch::Tensor x);
};

class resnet_model : public torch::nn::Module { 
    private: 
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::MaxPool2d maxpool{nullptr};
    torch::nn::Conv2d identity1{nullptr}, identity2{nullptr}, identity3{nullptr};
    torch::nn::Sequential layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
    torch::nn::Linear fc{nullptr};
    
    public:
    resnet_model();
    torch::Tensor forward(torch::Tensor x);
    
};

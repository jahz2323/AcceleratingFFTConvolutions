#include "AlexNet.hpp"

using namespace torch;

AlexNet::AlexNet() {
    // Use Overlap pooling 

    // First Convolutional Layer
    conv1 = register_module("Conv1" , torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).padding(1)));
    bn1 = register_module("BatchNorm1", torch::nn::BatchNorm2d(64));
    maxpool1 = register_module("Maxpool1", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
    
    // Second Convolutional Layer
    conv2 = register_module("Conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 192, 3).padding(1)));
    bn2 = register_module("BatchNorm2", torch::nn::BatchNorm2d(192));
    maxpool2 = register_module("Maxpool2", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
    
    // Third Convolutional Layer
    conv3 = register_module("Conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 384, 3).padding(1)));
    // Fourth Convolutional Layer
    conv4 = register_module("Conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).padding(1)));
    // Fifth Convolutional Layer
    conv5 = register_module("Conv5", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
    maxpool3 = register_module("Maxpool3", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));

    // Fully Connected Layers
    fc1 = register_module("FC1", torch::nn::Linear(256 * 4 * 4, 4096));
    fc2 = register_module("FC2", torch::nn::Linear(4096, 4096));
}
torch::Tensor AlexNet::forward(torch::Tensor x){
    x = torch::relu(conv1->forward(x));
    x = maxpool1->forward(x);
    x = bn1->forward(x);

    x = torch::relu(conv2->forward(x));
    x = maxpool2->forward(x);
    x = bn2->forward(x);

    x = torch::relu(conv3->forward(x));
    x = torch::relu(conv4->forward(x));
    x = torch::relu(conv5->forward(x));
    x = maxpool3->forward(x);
    
    x = x.view({x.size(0), -1}); // Flatten
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    x = torch::relu(fc2->forward(x));
    x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    return x;
}


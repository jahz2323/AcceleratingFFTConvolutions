#include "resnet.hpp"

Residual_Block::Residual_Block(int in_channels, int out_channels, int stride, int filter_size) {
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, filter_size).stride(stride).padding(1)));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, filter_size).stride(1).padding(1)));
    bn1 = register_module("batchnorm1", torch::nn::BatchNorm2d(out_channels));
    bn2 = register_module("batchnorm2", torch::nn::BatchNorm2d(out_channels));
    /* shortcut path */
    if(stride !=1 || in_channels != out_channels){
        shortcut = register_module("shortcut", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride)));
        shortcut_bn = register_module("shortcut_bn", torch::nn::BatchNorm2d(out_channels));
    }
    relu1 = register_module("relu1", torch::nn::ReLU());
    relu2 = register_module("relu2", torch::nn::ReLU());


}

torch::Tensor Residual_Block::forward(torch::Tensor x){
    auto identity = x; 

    x = conv1->forward(x);
    x = bn1->forward(x);
    x = relu1->forward(x);
    x = conv2->forward(x);
    x = bn2->forward(x);

    if (!shortcut.is_empty()){
        identity = shortcut->forward(identity);
        identity = shortcut_bn->forward(identity);
    }
    x += identity;
    x = relu2->forward(x);
    return x;
}

torch::Tensor resnet_model::forward(torch::Tensor x){
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);
    // x = maxpool->forward(x);

    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);

    //Global Average Pooling and FC
    x = torch::adaptive_avg_pool2d(x, {1,1});
    x = x.view({x.size(0), -1}); // Flatten
    x = fc->forward(x);
    return x.view({-1, 21, 3});;
}

resnet_model::resnet_model() {
    conv1 = register_module("Conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).padding(1)));
    bn1 = register_module("BatchNorm1", torch::nn::BatchNorm2d(64));
    // maxpool = register_module("MaxPool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));
    // Define layers using Residual Blocks
    layer1 = register_module("Layer1", torch::nn::Sequential(
        std::make_shared<Residual_Block>(64, 64, 2, 3), /*inchannels , out channels , stride, filter dim */
        std::make_shared<Residual_Block>(64, 64, 1, 3),
        std::make_shared<Residual_Block>(64, 64, 1, 3)
    ));
    conv2 = register_module("Conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 1)));
    identity1 = register_module("Identity1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 1)));
    layer2 = register_module("Layer2", torch::nn::Sequential(
        std::make_shared<Residual_Block>(64,  128, 2, 3),
        std::make_shared<Residual_Block>(128, 128, 1, 3), 
        std::make_shared<Residual_Block>(128, 128, 1, 3)
    ));
    identity2 = register_module("Identity2", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 1)));
    layer3 = register_module("Layer3", torch::nn::Sequential(
        std::make_shared<Residual_Block>(128, 256, 2, 3),
        std::make_shared<Residual_Block>(256, 256, 1, 3),
        std::make_shared<Residual_Block>(256, 256, 1, 3),
        std::make_shared<Residual_Block>(256, 256, 1, 3)
    ));
    identity3 = register_module("Identity3", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 1)));
    layer4 = register_module("Layer4", torch::nn::Sequential(
        std::make_shared<Residual_Block>(256, 512, 2, 3),
        std::make_shared<Residual_Block>(512, 512, 1, 3),
        std::make_shared<Residual_Block>(512, 512, 1, 3)
    ));
    fc = register_module("FC", torch::nn::Linear(512, 63)); // Keypoint for hands 
}

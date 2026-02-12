#include <iostream>
#include <string>
#include "inference.hpp"
#include "train.hpp"

int main(int argc, char* argv[]){
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mode> [additional arguments]" << std::endl;
        std::cerr << "Modes:" << std::endl;
        std::cerr << "  train_cifar" << std::endl;
        std::cerr << "  train_resnet" << std::endl;
        std::cerr << "  TestCustomOperator" << std::endl;
        std::cerr << "  inference <model_path> <data_path> <label_file_path>" << std::endl;
        return 1;
    }
    std::string mode = argv[1];
    if (mode == "train_cifar") {
        train::trainCIFAR();
    } 
    if (mode == "inference_cifar") {
        if (argc != 5) {
            std::cerr << "Usage for inference: " << argv[0] << " inference <model_path> <data_path> <label_file_path>" << std::endl;
            return 1;
        }
        std::string model_path = argv[2];
        inference::runInference(model_path);
    }
    if (mode == "train_resnet") {
        train::trainResNet();
    } 
    if (mode == "TestCustomOperator"){
        train::TestCustomOperator();
    }
    return 0;
}
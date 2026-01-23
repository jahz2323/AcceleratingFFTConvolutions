#include "inference.hpp"

void inference::runInference(const std::string& model_path, const std::string& data_path, const std::string& full_label_file_path){
    torch::Device device(torch::kCUDA);
    auto model = std::make_shared<AlexNet>();
    // load model
    torch::load(model,model_path);
    model->to(device);
    model->eval();

    // data is located 3 directories up from current path
    

    // load inference data 
    torch::Tensor input_tensor = DataHandling::LoadImageToTensor(data_path, device);

    // run inference
    auto output = model->forward(input_tensor);
    auto probabilities = torch::softmax(output, 1);

    std::cout << "Inference output probabilities: " << probabilities << std::endl;
    
    // get top-5 
    auto top5 = std::get<1>(probabilities.topk(5, 1, true, true));
    std::cout << "Top-5 predictions (class indices): " << top5 << std::endl;
}

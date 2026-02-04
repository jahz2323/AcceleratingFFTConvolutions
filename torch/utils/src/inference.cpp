#include "inference.hpp"

void inference::runInference(const std::string& model_path){
    // data folder is located 3 directories up from current path
    std::filesystem::path project_root = std::filesystem::current_path().parent_path().parent_path().parent_path();

    // label file: data/cifar/cifar-10-labels.txt
    CIFAR dataset = train::TestReadingCIFARBin(project_root.string(), CIFAR::Mode::TEST);
    std::filesystem::path label_file_path = project_root / "data" / "cifar" / "cifar-10-labels.txt";
    
    torch::Device device(torch::kCUDA);
    auto model = std::make_shared<AlexNet>();
    // load model
    torch::load(model,model_path);
    if (!model) {
            std::cerr << "Failed to load model from " << model_path << std::endl;
            return;
    }
    model->to(device);
    model->eval();

    
    auto example = dataset.get(rand() % dataset.size().value());

    auto input_tensor = example.data.to(device).unsqueeze(0); // add batch dimension
    auto target_label = example.target.item<int64_t>();
   

    // run inference on sample input tensor
    auto output = model->forward(input_tensor);
    auto probabilities = torch::nn::functional::softmax(output, 1);
    std::cout << "Inference output probabilities: " << probabilities << std::endl;
    std::cout << "Ground truth label for inference sample: " << target_label << std::endl;
    
    // get top-5 
    auto top5 = std::get<1>(probabilities.topk(5, 1, true, true));
    std::cout << "Top-5 predictions (class indices): " << top5 << std::endl;
}

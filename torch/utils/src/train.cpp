#include "train.hpp"

HandKeypoint train::KeypointFunction(const std::string &DATAPATH="/data/hand_dataset/hand_keypoint_dataset_26k/hand_keypoint_dataset_26k/"){ 
    std::filesystem::path curr_path = std::filesystem::current_path();
    std::cout << "Current path: " << curr_path << std::endl;
    /* Go back 3 dir and move to /hand_keypoint_dataset_26k/hand_keypoint_dataset_26k/ */
    std::filesystem::path root = curr_path.parent_path().parent_path().parent_path();
    std::filesystem::path data_dir = root.string() + DATAPATH;
    /* Print out data_dir */
    std::cout << "Data directory: " << data_dir << std::endl;
    /* Define img_dir and train_label_dir */
    std::string train_img_dir = data_dir.string() + "images/train";
    std::string train_label_dir = data_dir.string() + "labels/train";

    
    HandKeypoint HandKeypoint(train_img_dir, train_label_dir, false); // keep rgb images
    return HandKeypoint;
}

void train::TestCustomOperator(){
    void* handle = dlopen("./customops/libcustomops.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        std::cerr << "Failed to load library: " << dlerror() << std::endl;
        return;
    }
    std::vector<float> input_data = {
        1, 2, 3, 0,
        0, 1, 2, 3,
        3, 0, 1, 2,
        2, 3, 0, 1
    };
    std::vector<float> filter_data = {
        1, 0,
        0, 1
    };
    std::vector<float> expected_output_data = {
        2, 4, 6,
        0, 2, 4,
        6, 0, 2
    };
    torch::Device device(torch::kCUDA);
    torch::Tensor test_tensor = torch::from_blob(input_data.data(), {1, 1, 4, 4}, torch::kFloat).to(device);
    torch::Tensor filter_tensor = torch::from_blob(filter_data.data(), {1,1,2,2}, torch::kFloat).to(device);
    torch::Tensor expected_output = torch::from_blob(expected_output_data.data(), {1, 1, 3, 3}, torch::kFloat).to(device);
    std::cout << "test_tensor: " << test_tensor << std::endl;
    std::cout << "filter_tensor: " << filter_tensor << std::endl;

    int8_t stride = 1;
    int8_t padding = 0;
    // std::cout << "Test tensor: " << test_tensor << std::endl;
    auto op = torch::Dispatcher::singleton()
                .findSchemaOrThrow("my_ops::custom_2DConv", "")
                .typed<torch::Tensor (torch::Tensor, torch::Tensor, int8_t, int8_t)>();
    
    torch::Tensor conv_output = op.call(test_tensor, filter_tensor, stride, padding);
    torch::cuda::synchronize();
    std::cout << "Output of custom 2D convolution operator: " << conv_output << std::endl;
    std::cout << "Expected output: " << expected_output << std::endl;
    if (torch::allclose(conv_output, expected_output)) {
        std::cout << "Custom 2D convolution operator test passed!" << std::endl;
    } else {
        std::cout << "Custom 2D convolution operator test failed!" << std::endl;
    }
    {
        std::cout << "Testing custom autograd function for 2D convolution..." << std::endl;
        // test grad function 
        torch::Tensor input = torch::randn({1, 1, 4, 4}, torch::kCUDA).requires_grad_(true); 
        torch::Tensor weight = torch::randn({1, 1, 2, 2}, torch::kCUDA).requires_grad_(true); // we want to update weights wrt loss, so requires grad true
        //torch::Tensor bias = torch::randn({1}, torch::kCUDA).requires_grad_(true);
        int8_t stride = 2;
        int8_t padding = 0;
        auto y = myConv2DFunction::apply(input, weight, stride, padding);

        // create true output for loss calculation
        auto true_output = torch::randn_like(y);
        std::cout << "True output for loss calculation: " << true_output << std::endl;
        auto z = torch::nn::functional::mse_loss(y, true_output);

        std::cout << "Output of custom autograd convolution function: " << y << std::endl;
        std::cout << "Mean of output (loss): " << z.item<float>() << std::endl;
        z.backward();

        std::cout << "Input gradient: " << input.grad() << std::endl;
        std::cout << "Weight gradient: " << weight.grad() << std::endl;

        //update weights with simple SGD step
        float learning_rate = 0.01;
        weight.data().sub_(learning_rate * weight.grad());
        std::cout << "Updated weights after one SGD step: " << weight << std::endl;

        // TESTING TRAINING
        struct SimpleCNN : torch::nn::Module {
            SimpleCNN() {
                w1 = register_parameter("w1", torch::randn({8, 1, 5, 5}, torch::kCUDA).requires_grad_(true)); // OutputChannels,InputChannels,Kernelw,Kernelh
                w2 = register_parameter("w2", torch::randn({5, 8, 2, 2}, torch::kCUDA).requires_grad_(true)); // second conv layer weights for testing multiple layers
            }
            torch::Tensor forward(torch::Tensor x) {
                x = myConv2DFunction::apply(x, w1, 1, 0); // valid conv with stride 1 -> input dims 16x16, filter 5x5 -> output dims 8x12x12
                x = torch::relu(x);
                x = myConv2DFunction::apply(x, w2, 1, 0); // second conv layer with 32 filters of size 2x2, input dims 8x12x12 -> output dims 32x11x11
                x = torch::relu(x);
                return x;
            }
            torch::Tensor w1, w2;
        };
        SimpleCNN model;
        model.to(torch::kCUDA);
        torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.01));
        torch::Tensor input_data = torch::randn({1, 1, 16, 16}, torch::kCUDA); // batch=32, channels=1, width=4, height=4
        auto output_for_shape = model.forward(input_data);
        std::cout << "Output shape from model forward pass: " << output_for_shape.sizes() << std::endl;
        auto target = torch::randn_like(output_for_shape);

        for (int epoch = 0; epoch < 10; ++epoch) {
            optimizer.zero_grad();
            auto output = model.forward(input_data);
            auto loss = torch::nn::functional::mse_loss(output, target);
            std::cout << "Epoch " << epoch << ", Loss: " << loss.item<float>() << std::endl;
            loss.backward();
            optimizer.step();
            optimizer.zero_grad();
        }
        
        // After training, check if weights have been updated
        std::cout << "Final weights after training: W1: " << model.w1 << "W2:" << model.w2 << std::endl;
    }
    
}


/**
    Function to train AlexNet model on CIFAR dataset
*/
void train::trainCIFAR(){
    const size_t BatchSize = 128;
    const size_t Epochs = 100;
    const size_t numworkers = 4; // number of data loading workers
    const float learning_rate = 0.01;
    const float momentum = 0.9;
    
    std::filesystem::path root = std::filesystem::current_path().parent_path().parent_path().parent_path();

    torch::Device device(torch::kCUDA);
    auto model = std::make_shared<AlexNet>();

    auto train_dataset = (CIFAR(root.string(), CIFAR::Mode::TRAIN)).map(torch::data::transforms::Stack<>()); // no normalization steps atm
    auto test_dataset = (CIFAR(root.string(), CIFAR::Mode::TEST)).map(torch::data::transforms::Stack<>()); // no normalization steps atm

    auto num_train_samples = train_dataset.size().value();
    std::cout << "Number of training samples: " << num_train_samples << std::endl;
    auto num_test_samples = test_dataset.size().value();
    std::cout << "Number of test samples: " << num_test_samples << std::endl;
    
    auto dl = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(train_dataset), torch::data::DataLoaderOptions().batch_size(BatchSize).workers(numworkers));
    auto tl = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), torch::data::DataLoaderOptions().batch_size(BatchSize).workers(numworkers));
    // Optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(learning_rate).momentum(momentum));
    model->to(device);

    
    // Training loop
    for (size_t epoch = 1; epoch <= Epochs; ++epoch){
        size_t batch_index = 0;
        double running_loss = 0.0;
        size_t correct = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for(auto& batch: *dl){
            
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);

            // Forward pass
            auto output = model->forward(data);

            // Compute loss
            auto loss = torch::nn::functional::cross_entropy(output, targets);

            // update running loss
            running_loss += loss.item<double>() * data.size(0);

            // calculate prediction 
            auto pred = output.argmax(1);

            //update number of correct predictions
            correct += pred.eq(targets).sum().item<int64_t>();

            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epoch_duration = end - start;
        std::cout << "Epoch " << epoch << " completed in " << epoch_duration.count() << " seconds." << std::endl;
        double throughput = static_cast<double>(num_train_samples) / epoch_duration.count();
        std::cout << "Throughput: " << throughput << " samples/second" << std::endl;
        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<double>(correct) / num_train_samples * 100.0;
        std::cout << "Epoch: " << epoch << " | Average Loss: " << sample_mean_loss << " | Accuracy: " << accuracy << "%" << std::endl;
    }
    //Training finished 
    std::cout << "Training finished. Testing model..." << std::endl;
    //Testing model
    model->eval();
    size_t correct = 0;

    for (auto& batch: *tl){
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);

        auto output = model->forward(data);

        auto loss = torch::nn::functional::cross_entropy(output, targets);
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().item<int64_t>();
    }
    std::cout << "Test Accuracy: " << static_cast<double>(correct) / num_test_samples * 100.0 << "%" << std::endl;

    auto test_accuracy = static_cast<double>(correct) / num_test_samples * 100.0;
    auto test_sample_mean_loss = static_cast<double>(correct) / num_test_samples;

    std::cout << "Test completed. Accuracy: " << test_accuracy << "%, Average Loss: " << test_sample_mean_loss << std::endl;
    
    // save model 
    torch::save(model, "alexnet_cifar10_model.pt");
}

/**
    @brief Training function for ResNet model on Hand Keypoint Dataset
*/
void train::trainResNet(){
    HandKeypoint raw_dataset = train::KeypointFunction();
    const size_t BatchSize = 32;
    const size_t Epochs = 20;
    const size_t numworkers = 4; // number of data loading workers
    const float learning_rate = 0.001;
    const float momentum = 0.9;
    torch::Device device(torch::kCUDA);
    auto model = std::make_shared<resnet_model>();
    auto stacked_dataset = raw_dataset.map(torch::data::transforms::Stack<>());
    auto dl = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(stacked_dataset), 
        torch::data::DataLoaderOptions().batch_size(BatchSize).workers(numworkers)
    );
    // Optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions
    (learning_rate).momentum(momentum));
    model->to(device);

    std::cout << "Starting training ResNet model on Hand Keypoint Dataset..." << std::endl;
    // Training loop
    for (size_t epoch = 1; epoch <= Epochs; ++epoch){
        size_t batch_index = 0;
        double running_loss = 0.0;
        auto start = std::chrono::high_resolution_clock::now();
        for(auto& batch: *dl){
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);
            // Forward pass
            auto output = model->forward(data);

            // Compute loss - Regressive task
            auto loss = torch::nn::functional::mse_loss(output, targets);
            // update running loss
            running_loss += loss.item().toFloat() * data.size(0);
            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epoch_duration = end - start;
        std::cout << "Epoch " << epoch << " completed in " << epoch_duration.count() << " seconds." << std::endl;
        
        auto sample_mean_loss = running_loss / raw_dataset.size().value();
        std::cout << "Epoch: " << epoch << " | Average Loss: " << sample_mean_loss << std::endl;
    }
    //Training finished 
    std::cout << "Training finished." << std::endl;
    // save model 
    torch::save(model, "resnet_handkeypoint_model.pt");

}

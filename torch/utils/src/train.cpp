#include "train.hpp"

/**
    @brief Function to test HandKeypoint loading and display first sample

*/
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

    
    HandKeypoint HandKeypoint(train_img_dir, train_label_dir, false);

    std::cout << "Dataset length: " << HandKeypoint.size().value() << std::endl;

    HandSample HandSample = HandKeypoint.getItem(1);
    std::cout << "First HandSample image size: " << HandSample.image.size() << std::endl;

    int img_w = HandSample.image.cols;
    int img_h = HandSample.image.rows;

    // convert keypoints tensor to vector for display
    cv::Mat keypoints = DataHandling::torchTensortoCVMat(HandSample.keypoints);
    std::cout << "First HandSample keypoints: " << keypoints << std::endl;

    //plot bounding box 

    float x = HandSample.bounding_box[1].item<float>() * img_w - (HandSample.bounding_box[3].item<float>() * img_w) / 2.0;
    float y = HandSample.bounding_box[2].item<float>() * img_h - (HandSample.bounding_box[4].item<float>() * img_h) / 2.0;
    float w = HandSample.bounding_box[3].item<float>() * img_w;
    float h = HandSample.bounding_box[4].item<float>() * img_h;
    cv::rectangle(HandSample.image, 
    cv::Rect(static_cast<int>(x), static_cast<int>(y), static_cast<int>(w), static_cast<int>(h)), 
    cv::Scalar(255, 0, 0), 2
    );


    // map keypoints onto image for visualization
    for (int i = 0; i < keypoints.rows; i++) {
        float x = keypoints.at<float>(i, 0) * img_w;
        float y = keypoints.at<float>(i, 1) * img_h;
        int visibility = static_cast<int>(keypoints.at<float>(i, 2));

        if (visibility >= 0) { // only plot visible keypoints
            cv::circle(HandSample.image, cv::Point(static_cast<int>(x), static_cast<int>(y)), 3, cv::Scalar(0, 255, 0), -1);
        }
    }

    DataHandling::displayImage(HandSample.image, "First HandSample Image");

    std::cout << "Press any key to continue..." << std::endl;
    char k = cv::waitKey(0);
    std::cout << "Returning HandKeypoint dataset object." << std::endl;
    return HandKeypoint;
}

void train::TestCustomOperator(){
    // void* handle = dlopen("./libtorchapp_ops.so", RTLD_NOW | RTLD_GLOBAL);
    // if (!handle) {
    //     std::cerr << "Failed to load library: " << dlerror() << std::endl;
    //     return 1;
    // }

    // torch::Tensor test_tensor = torch::randint(0, 10, {100}, torch::kCUDA).to(torch::kInt);
    // // std::cout << "Test tensor: " << test_tensor << std::endl;
    // auto op = torch::Dispatcher::singleton()
    //             .findSchemaOrThrow("my_ops::custom_allreduce", "")
    //             .typed<torch::Tensor (torch::Tensor)>();
    
    // torch::Tensor reduced_sum = op.call(test_tensor);
    // torch::cuda::synchronize();
    // std::cout << "Reduced sum: " << reduced_sum << std::endl;
}
/**
    @brief Sequential data loading for CIFAR#
    TODO: Parallelize data loading using multithreading 
*/
CIFAR train::TestReadingCIFARBin(std::string root, CIFAR::Mode mode){
    std::string cifar_bin_path = "data/cifar/cifar-10-batches-bin/data_batch_1.bin";
    std::string full_path = root + "/" + cifar_bin_path;
    CIFAR dataset(mode);
    
    // DataHandling::load_binary_file(full_path);
    // torch::data::Example<> example = dataset.get(0); // 2 tensors: image and label
    // // print out size of dataset and example image and label in tensor form
    // std::cout << "CIFAR Dataset size: " << dataset.size().value() << std::endl;
    // std::cout << "Example image tensor size: " << example.data.sizes() << std::endl;
    // std::cout << "Example label tensor value: " << example.target << std::endl;
    
    // For each bin file in /train - read in and save to dataset object 

    //start time 
    clock_t start_time = clock();
    CIFARdata cifar_data;
    for (const auto &entry: fs::directory_iterator(root + "/data/cifar/cifar-10-batches-bin/")){
        // if mode is train - only read train files
        if (mode == CIFAR::Mode::TRAIN && entry.path().string().find("data_batch") != std::string::npos){
            std::cout << "Loading CIFAR binary file: " << entry.path().string() << std::endl;
            cifar_data = DataHandling::load_binary_file(entry.path().string());
        }
        // if mode is test - only read test file
        else if (mode == CIFAR::Mode::TEST && entry.path().string().find("test_batch") != std::string::npos){
            std::cout << "Loading CIFAR binary file: " << entry.path().string() << std::endl;
            cifar_data = DataHandling::load_binary_file(entry.path().string());
        }
    }

    // end time
    clock_t end_time = clock();
    double elapsed_time = double(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "Time taken to load all CIFAR binary files: " << elapsed_time << " seconds" << std::endl;
    /**
    @note 5 seconds in total to load all 5 CIFAR bin files (~10,000 images each) on CPU
    */
    // check total size of dataset 
    std::cout << "Total CIFAR Dataset size after loading all bin files: " << cifar_data.images.size() << std::endl;
    
    // pass data to dataset object
    dataset.setData(cifar_data.images, cifar_data.labels);
    return dataset;
}

/* Testing Dataset loading */
void train::trainCIFAR(){
    const size_t BatchSize = 128;
    const size_t Epochs = 100;
    const size_t numworkers = 4; // number of data loading workers
    const float learning_rate = 0.01;
    const float momentum = 0.9;
    
    std::filesystem::path root = std::filesystem::current_path().parent_path().parent_path().parent_path();

    torch::Device device(torch::kCUDA);
    auto model = std::make_shared<AlexNet>();

    auto train_dataset = (train::TestReadingCIFARBin(root.string(), CIFAR::Mode::TRAIN)).map(torch::data::transforms::Stack<>()); // no normalization steps atm
    auto test_dataset = (train::TestReadingCIFARBin(root.string(), CIFAR::Mode::TEST)).map(torch::data::transforms::Stack<>()); // no normalization steps atm

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


void train::trainResNet(){
    HandKeypoint HandKeypoint = train::KeypointFunction();
    const size_t BatchSize = 32;
    const size_t Epochs = 50;
    const size_t numworkers = 4; // number of data loading workers
    const float learning_rate = 0.001;
    const float momentum = 0.9;
    torch::Device device(torch::kCUDA);
    auto model = std::make_shared<resnet_model>();
    auto dl = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(HandKeypoint), torch::data::DataLoaderOptions().batch_size(BatchSize).workers(numworkers));
    // Optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions
    (learning_rate).momentum(momentum));
    model->to(device);
    // Training loop
    for (size_t epoch = 1; epoch <= Epochs; ++epoch){
        size_t batch_index = 0;
        double running_loss = 0.0;
        auto start = std::chrono::high_resolution_clock::now();
        for(auto& batch: *dl){
            auto data = batch.image.to(device);
            auto targets = batch.keypoints.to(device);
            // Forward pass
            auto output = model->forward(data);

            // Compute loss
            auto loss = torch::nn::functional::mse_loss(output, targets);
            // update running loss
            running_loss += loss.item<double>() * data.size(0);
            // Backward pass and optimize
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epoch_duration = end - start;
        std::cout << "Epoch " << epoch << " completed in " << epoch_duration.count() << " seconds." << std::endl;
        double throughput = static_cast<double>(HandKeypoint.size().value()) / epoch_duration.count();
        std::cout << "Throughput: " << throughput << " samples/second" << std::endl;
        auto sample_mean_loss = running_loss / HandKeypoint.size().value();
        std::cout << "Epoch: " << epoch << " | Average Loss: " << sample_mean_loss << std::endl;
    }
    //Training finished 
    std::cout << "Training finished." << std::endl;
    // save model 
    torch::save(model, "resnet_handkeypoint_model.pt");

}
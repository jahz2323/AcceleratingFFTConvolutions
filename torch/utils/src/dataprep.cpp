#include "dataprep.hpp"


cv::Mat DataHandling::torchTensortoCVMat(const torch::Tensor& tensor){
    // Ensure tensor is on CPU and of type float32
    torch::Tensor cpu_tensor = tensor.to(torch::kCPU).to(torch::kFloat32);
    // Get dimensions
    auto sizes = cpu_tensor.sizes();
    int rows = sizes[0];
    int cols = sizes[1];
    // Create cv::Mat
    cv::Mat mat(rows, cols, CV_32FC1, cpu_tensor.data_ptr<float>());
    return mat.clone(); // return a clone to ensure data safety
}


/**
    @brief Helper to load image with opencv and convert to torch::Tensor, on device if available 
    @param img_path: path to image file
    @param device: torch::Device to load tensor onto
    @return torch::Tensor image tensor
*/
torch::Tensor DataHandling::LoadImageToTensor(const std::string& img_path, torch::Device device){
    cv::Mat img = cv::imread(img_path);
    if(!img.data || img.empty()){
        std::cerr << "Could not open or find the image at path: " << img_path << std::endl;
        throw std::runtime_error("Image not found at path: " + img_path);
    }
    // Convert BGR to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(32,32)); // resize to 32x32 for CIFAR 
    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kUInt8);
    img_tensor = img_tensor.permute({2,0,1}); // change to CxHxW
    img_tensor = img_tensor.to(torch::kFloat32).div(255.0).unsqueeze(0); // normalize to [0,1] and add batch dimension
    img_tensor = img_tensor.to(device);
    return img_tensor;
}
 /**
    * @brief Display image using OpenCV
    * @param img: cv::Mat image to display
    * @param window_name: name of the display window
    * @param desired_width: desired width of the display window
    * @param desired_height: desired height of the display window
    * @param keep_aspect_ratio: whether to keep the aspect ratio of the image
    * @return void
    */
void DataHandling::displayImage(cv::Mat img, const std::string& window_name, int desired_width, int desired_height, bool keep_aspect_ratio) {
    // Create a window for display.
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);

    // resize window
    if (keep_aspect_ratio) {
        int original_width = img.cols;
        int original_height = img.rows;
        float aspect_ratio = static_cast<float>(original_width) / static_cast<float>(original_height);

        if (desired_width / aspect_ratio <= desired_height) {
            desired_height = static_cast<int>(desired_width / aspect_ratio);
        } else {
            desired_width = static_cast<int>(desired_height * aspect_ratio);
        }
    }
    cv::resizeWindow(window_name, desired_width, desired_height);

    cv::imshow(window_name, img);
    int k = cv::waitKey(0);
    if (k == 's') {
        cv::imwrite(window_name + ".png", img);
    }
    cv::destroyWindow(window_name);
}


EMNISTDataset::EMNISTDataset(const std::string& image_path, const std::string& label_path) {
// 1. Read Images
    std::ifstream img_file(image_path, std::ios::binary);
    if(!img_file.is_open()){
        throw std::runtime_error("could not open image file at: " + image_path);
    }

    // 2. read indexes 
    int32_t magic_number = read_int(img_file);
    int32_t num_items = read_int(img_file); 
    int32_t rows = read_int(img_file);
    int32_t cols = read_int(img_file); 

    // Debug output to verify validitiy of numbers 
    std::cout << "Data check: Magic=" << magic_number << " Items=" << num_items 
                << " Dim=" << rows << "x" << cols << std::endl;

    if (num_items <= 0 || rows <= 0 || cols <= 0) {
        throw std::runtime_error("Invalid IDX header values. Check if file is unzipped.");
    }

    // The overflow happened here: (num_items * rows * cols)
    size_t total_pixels = static_cast<size_t>(num_items) * rows * cols;
    std::vector<uint8_t> img_buffer(total_pixels);
    img_file.read(reinterpret_cast<char*>(img_buffer.data()), total_pixels);
    
    images_ = torch::from_blob(img_buffer.data(), {num_items, 1, rows, cols}, torch::kUInt8)
                .to(torch::kFloat32).div(255.0).clone();

    // 2. Read Labels
    std::ifstream lbl_file(label_path, std::ios::binary);
    if (!lbl_file.is_open()) {
        throw std::runtime_error("Could not open label file: " + label_path);
    }

    magic_number = read_int(lbl_file);
    num_items = read_int(lbl_file);

    std::vector<uint8_t> lbl_buffer(num_items);
    lbl_file.read(reinterpret_cast<char*>(lbl_buffer.data()), num_items);
    
    labels_ = torch::from_blob(lbl_buffer.data(), {num_items}, torch::kUInt8)
                .to(torch::kInt64).clone();

    std::cout << "Successfully loaded " << num_items << " samples." << std::endl;
}


/* Source file to load Keypoint dataset*/
HandKeypoint::HandKeypoint (std::string img_directory, std::string label_directory, bool apply_transform) : img_dir(img_directory), label_dir(label_directory), transform(apply_transform) {
    // Populate file_names vector with image file names from img_dir
    for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
        img_file_names.push_back(entry.path().filename().string());
    }
}
torch::data::Example<> HandKeypoint::get(size_t index) {
    // Load image
    std::string img_path = img_dir + "/" + img_file_names[index];
    cv::Mat img = cv::imread(img_path);

   //Safe pathing - 
   std::string filename = img_file_names[index];
   size_t last_dot = filename.find_last_of(".");

   // Create stem of filename
   std::string stem = (last_dot == std::string::npos) ? filename : filename.substr(0, last_dot);
   std::string label_path = label_dir + "/" + stem + ".txt";
   
    std::ifstream label_file(label_path);
    
    // store as torch::Tensor target
    torch::Tensor image_tensor;
    torch::Tensor bounding_box;
    torch::Tensor keypoints;
    if (label_file.is_open()) {
        std::shared_ptr<std::vector<float>> keypoint_values = std::make_shared<std::vector<float>>();
        float value;
        while (label_file >> value) {
            keypoint_values->push_back(value);
        }
        /* Convert list of keypoint values to torch tensor 
            torch::Tensor bounding_box = [<class-index> <x> <y> <width> <height>]
            torch::Tensor keypoints = [(skip first 4), {21 keypoints: (x, y, visibility)}]
            class, centre_x, centre_y, width, height, kpx, pky visibility... kpx_n, kpy_n, visibility_n 
        */
        bounding_box = torch::from_blob(keypoint_values->data(), {5}, torch::kFloat32).clone(); 
        keypoints = torch::from_blob(keypoint_values->data()+5, {21, 3}, torch::kFloat32).clone();
        
        /* Print out */
        //std::cout << "Bounding box tensor: " << bounding_box << std::endl;
        //std::cout << "Keypoints tensor: " << keypoints << std::endl;

        label_file.close(); // close the file after reading
    } else {
        throw std::runtime_error("Could not open label file: " + label_path);
    }
    
    // Apply transformations if needed
    if (transform) {
        // Example transformation: convert to grayscale
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }
    // Convert cv::Mat to torch::Tensor
    image_tensor = torch::from_blob(img.data, {img.rows, img.cols, img.channels()}, torch::kUInt8);
    image_tensor = image_tensor.permute({2,0,1}); // change to CxHxW
    image_tensor = image_tensor.to(torch::kFloat32).div(255.0); // normalize to [0,1]
    
   
    return {image_tensor, keypoints};

};
/**
    @brief Sequential data loading for CIFAR#
    TODO: Parallelize data loading using multithreading 
*/
CIFAR CIFAR::TestReadingCIFARBin(std::string root, CIFAR::Mode mode){
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

/**
    @brief Load CIFAR binary file and populate dataset
    @param path_dir: path to CIFAR binary file
    return tensor containing std::vector of images and labels
*/
CIFARdata DataHandling::load_binary_file(const std::string& path_dir){
    fs::path dir_path(path_dir);
    if(!fs::exists(dir_path)){
        throw std::runtime_error("Directory does not exist: " + path_dir);
    }
    std::streampos begin,end; 
    std::ifstream file(dir_path.string(), std::ios::in | std::ios::binary);
    if(!file.is_open()){
        throw std::runtime_error("Could not open file: " + dir_path.string());
    }
    begin = file.tellg();
    file.seekg(0, std::ios::end);
    end = file.tellg();
    size_t file_size = end - begin;
    size_t num_entries = file_size / sizeof(CIFARBuffer);
    std::cout << "Actual size of CIFARBuffer: " << sizeof(CIFARBuffer) << " bytes" << std::endl;
    std::cout << "Number of entries in CIFAR binary file: " << num_entries << std::endl;

    // Reset pointer to beginning
    file.seekg(0, std::ios::beg);
    CIFARdata cifar_data;

    for(size_t i=0; i<num_entries; i++){
        CIFARBuffer buffer = {0};
        if(file.read(reinterpret_cast<char*>(&buffer), sizeof(CIFARBuffer))){
            std::cout << "Read entry " << i << " with label: " << static_cast<int>(buffer.label) << std::endl;
        }
        else{
            throw std::runtime_error("Error reading entry " + std::to_string(i)+ "Bytes read: " + std::to_string(file.gcount()));
        }
        // Comnvert Uint8_t data to torch::Tensor
        torch::Tensor img_tensor = torch::from_blob(buffer.data, {3,32,32}, torch::kUInt8).to(torch::kFloat32).div(255.0).clone();
        torch::Tensor label_tensor = torch::tensor(static_cast<int64_t>(buffer.label), torch::kInt64);
        // Append to CIFARdata struct
        
        cifar_data.images.push_back(img_tensor);
        cifar_data.labels.push_back(label_tensor);
    }
    file.close();
    
    return cifar_data;
}


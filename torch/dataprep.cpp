#include "dataprep.hpp"



/* Source file to load Keypoint dataset*/
KeypointDataset::KeypointDataset (std::string img_directory, std::string label_directory, bool apply_transform) : img_dir(img_directory), label_dir(label_directory), transform(apply_transform) {
    // Populate file_names vector with image file names from img_dir
    for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
        file_names.push_back(entry.path().filename().string());
    }
}

HandSample KeypointDataset::get(size_t index) {
    // Load image
    std::string img_path = img_dir + "/" + file_names[index];
    cv::Mat img = cv::imread(img_path);

    // Load keypoints
    auto start_pos = file_names[index].find(".");
    std::string label_path = label_dir + "/" + file_names[index].erase(start_pos, start_pos + 3) + ".txt";
    // std::cout << "Loading image: " << img_path << " and label: " << label_path << std::endl;
    std::ifstream label_file(label_path);
    
    // store as torch::Tensor target
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
        std::cout << "Bounding box tensor: " << bounding_box << std::endl;
        std::cout << "Keypoints tensor: " << keypoints << std::endl;

        label_file.close(); // close the file after reading
    } else {
        throw std::runtime_error("Could not open label file: " + label_path);
    }
    
    // Apply transformations if needed
    if (transform) {
        // Example transformation: convert to grayscale
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }

   
    return HandSample{img, keypoints, bounding_box};


};
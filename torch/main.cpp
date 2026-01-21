#include <torch/torch.h>
#include <torch/script.h>
#include <torch/custom_class.h>
#include <torch/library.h>
#include <opencv2/opencv.hpp> 
#include <ATen/cuda/CUDAContext.h>
#include <dlfcn.h>
#include "dataprep.hpp" 

#include "models/AlexNet.hpp"

 /**
 * @brief Display image using OpenCV
 * @param img: pointer to cv::Mat image to display
 * @param window_name: name of the display window
 * @param desired_width: desired width of the display window
 * @param desired_height: desired height of the display window
 * @param keep_aspect_ratio: whether to keep the aspect ratio of the image
 * @return void
  */

void displayImage(cv::Mat img, const std::string& window_name, int desired_width = 800, int desired_height = 600, bool keep_aspect_ratio = true) {
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
}

void KeypointFunction(){ 
    //std::filesystem::path curr_path = std::filesystem::current_path();
    //std::cout << "Current path: " << curr_path << std::endl;
    // /* Go back 3 dir and move to /hand_keypoint_dataset_26k/hand_keypoint_dataset_26k/ */
    // std::filesystem::path root = curr_path.parent_path().parent_path().parent_path();
    // std::filesystem::path data_dir = root.append(DATAPATH);
    // /* Print out data_dir */
    // std::cout << "Data directory: " << data_dir << std::endl;
    // /* Define img_dir and train_label_dir */
    // std::string train_img_dir = data_dir.string() + "images/train";
    // std::string train_label_dir = data_dir.string() + "labels/train";

    
    // KeypointDataset dataset(train_img_dir, train_label_dir, false);

    // std::cout << "Dataset length: " << dataset.size().value() << std::endl;

    // HandSample sample = dataset.getItem(0);
    // std::cout << "First sample image size: " << sample.image.size() << std::endl;
    
    // // convert keypoints tensor to vector for display
    // cv::Mat keypoints = dataset.torchTensortoCVMat(sample.keypoints);
    // std::cout << "First sample keypoints: " << keypoints << std::endl;

    //displayImage(sample.image, "First Sample Image");
}
void TestCustomOperator(){
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

void TestReadingCIFARBin(std::string root){
    std::string cifar_bin_path = "data/cifar/cifar-10-batches-bin/data_batch_1.bin";
    std::string full_path = root + "/" + cifar_bin_path;
    CIFAR dataset(full_path);
    std::cout << "CIFAR Dataset size: " << dataset.size().value() << std::endl;
    auto sample = dataset.get(0);
    std::cout << "First sample image tensor size: " << sample.data.sizes() << std::endl;
    std::cout << "First sample label tensor: " << sample.target << std::endl;
}

/* Testing Dataset loading */
int main(){

    std::filesystem::path root = std::filesystem::current_path().parent_path().parent_path().parent_path();

    torch::Device device(torch::kCUDA);
    auto model = std::make_shared<AlexNet>();
    
    TestReadingCIFARBin(root.string());
    return 0;
}
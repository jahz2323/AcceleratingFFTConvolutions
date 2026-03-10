#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp> 
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <vector>
#include <arpa/inet.h>
namespace fs = std::filesystem;

/**
 * @brief Structure to hold a single hand sample with image and keypoints
 * 
 */
struct HandSample  {
    cv::Mat image;
    torch::Tensor keypoints;
    torch::Tensor bounding_box;
};
/**
    * @brief Structure to hold CIFAR binary data
    * 
*/

#pragma pack(push,1)
struct CIFARBuffer{
    uint8_t label; 
    uint8_t data[3*32*32];
};
#pragma pack(pop)

struct CIFARdata{
    std::vector<torch::Tensor> images;
    std::vector<torch::Tensor> labels;
};

namespace DataHandling { 
    cv::Mat torchTensortoCVMat(const torch::Tensor& tensor);
    torch::Tensor LoadImageToTensor(const std::string& img_path, torch::Device device);
    void displayImage(cv::Mat img, const std::string& window_name, int desired_width = 800, int desired_height = 600, bool keep_aspect_ratio = true);
    std::vector<std::tuple<std::string, int64_t>> Read_Dir(const std::string& path_dir);
    CIFARdata load_binary_file(const std::string& path_dir); 
}

/**
 * @brief Dataset class to load Key
 TODO: Change to template torch::data::Example<>>
 */
class HandKeypoint : public torch::data::datasets::Dataset<HandKeypoint, torch::data::Example<>> {
private:  
    std::string img_dir;
    std::string label_dir;
    std::vector<std::string> img_file_names;
    bool transform;     
public: 
    HandKeypoint(std::string img_directory, std::string label_directory, bool apply_transform=false);
    torch::data::Example<> get(size_t index) override;
    inline torch::optional<size_t> size() const override {return img_file_names.size();}
    
};

/**
 * @brief Generic Dataset class template
   @author https://github.com/mhubii
 */
class CIFAR : public torch::data::datasets::Dataset<CIFAR, torch::data::Example<>> {
    public:
    enum class Mode{
        TRAIN,
        TEST
    };
    explicit CIFAR(Mode mode) : mode_(mode) {};
    CIFAR TestReadingCIFARBin(std::string root, CIFAR::Mode mode);
    inline torch::data::Example<> get(size_t index) override { return {images_[index], labels_[index]};}
    inline torch::optional<size_t> size() const override {return images_.size();};
    inline void setData(std::vector<torch::Tensor> images, std::vector<torch::Tensor> labels){
        images_ = images;
        labels_ = labels;
    }
    private: 
    std::vector<torch::Tensor> images_; 
    std::vector<torch::Tensor>  labels_;
    Mode mode_;
};
/**
    @brief Dataset class to load EMNIST dataset
*/
class EMNISTDataset : public torch::data::Dataset<EMNISTDataset> {
private: 
    torch::Tensor images_, labels_; 

    // helper function to flip endianness IDX are big endian 
    inline int32_t flip_endian(int32_t n){
        unsigned char ch1, ch2, ch3, ch4; 
        ch1 = n & 255; ch2 = (n >> 8) & 255; ch3 = (n >>16) & 255; ch4 = (n >> 24) & 255 ; 
        return ((int32_t)ch1 <<24 ) + ((int32_t)ch2 << 16) + ((int32_t)ch3 << 8) + ch4; 
    }

    inline int32_t read_int(std::ifstream& file){
        int32_t val = 0;
        file.read(reinterpret_cast<char*>(&val),4); 
        // ntohl converts 'network' (big-endian) to 'host' (little-endian)
        return ntohl(val);
    }

public: 
    EMNISTDataset(const std::string& image_path, const std::string& label_path);
    // Required by LibTorch
    inline torch::data::Example<> get(size_t index) override {
        return {images_[index], labels_[index]};
    }

    inline torch::optional<size_t> size() const override {
        return images_.size(0);
    }
};

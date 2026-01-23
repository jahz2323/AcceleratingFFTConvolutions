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

/**
    * @brief Methods to read directory and return list of file names and labels
    @func1:
    Read_Dir - reads CIFAR binary file and returns vector of CIFARBuffer structs
    @returns vector<CIFARBuffer>
    @func2:
    Read_Dir - reads directory of images and returns vector of tuples (file_name, label
    @returns vector<tuple<string, int64_t>>
*/
std::vector<std::tuple<std::string, int64_t>> Read_Dir(const std::string& path_dir);

namespace DataHandling { 
    torch::Tensor LoadImageToTensor(const std::string& img_path, torch::Device device);
    void displayImage(cv::Mat img, const std::string& window_name, int desired_width = 800, int desired_height = 600, bool keep_aspect_ratio = true);
}

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
    // Helper function to read CIFAR binary file
    inline void load_binary_file(const std::string& path_dir){
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
            images_.push_back(img_tensor);
            labels_.push_back(torch::tensor(static_cast<int64_t>(buffer.label), torch::kInt64));
        }
        file.close();
        
        // call save image function 
        std::cout << "Successfully loaded the file" << dir_path.string() <<"With size of " << images_.size() << " entries from CIFAR binary file." << std::endl;
    }
    //Constructer
    explicit CIFAR(Mode mode) : mode_(mode) {};

    inline torch::data::Example<> get(size_t index) override {
        return {images_[index], labels_[index]};
    }
    // Override the size method to infer the size of the data set.
    inline torch::optional<size_t> size() const override {
        return images_.size();
    };
    private: 
    std::vector<torch::Tensor> images_; 
    std::vector<torch::Tensor>  labels_;
    Mode mode_;
};

/**
 * @brief Dataset class to load Key
 */
class KeypointDataset : public torch::data::datasets::Dataset<KeypointDataset, HandSample> {
private:  
    std::string img_dir;
    std::string label_dir;
    std::vector<std::string> file_names;
    bool transform;     
public: 
    KeypointDataset(std::string img_directory, std::string label_directory, bool apply_transform=false);
    HandSample get(size_t index) override;
    inline torch::optional<size_t> size() const override {
        return file_names.size();
    }

    inline HandSample getItem(size_t index) {
        HandSample sample = get(index);
        return sample;
    }
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

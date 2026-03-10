#include "utils.cuh"

void utils::printGPUMemoryInfo(const std::string& label){
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if(err != cudaSuccess){
        std::cerr << "Error getting GPU memory info: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    std::cout << label << " - GPU Memory Info: Free: " << free_mem / (1024.0 * 1024.0) << " MB, Total: " << total_mem / (1024.0 * 1024.0) << " MB" << std::endl;
}

int utils::nextPowerOfTwo(int n){
    int count = 0;
    // First n in the below condition is for the case where n is 0
    if (n && !(n & (n - 1)))
        return n;

    while( n != 0){
        n >>= 1;
        count += 1;
    }
    return 1 << count;
}

/**
    @brief Measure the mean squared error between two float vectors.
    @param output1 The first output vector.
    @param output2 The second output vector.
    @return The mean squared error between the two vectors.
*/
float utils::MeasureError(const std::vector<float>& output1, const std::vector<float>& output2){
    if(output1.size() != output2.size()){
        throw std::invalid_argument("Output vectors must be of the same size to measure error.");
    }
    float mse = 0.0f;
    for(size_t i = 0; i < output1.size(); ++i){
        float diff = output1[i] - output2[i];
        mse += diff * diff;
    }
    mse /= static_cast<float>(output1.size());
    return mse;
}
bool utils::validateResults(const std::vector<float>& output1, const std::vector<float>& output2, float tolerance = 1e-1f){
    if(output1.size() != output2.size()){
        throw std::invalid_argument("Output vectors must be of the same size to validate results.");
    }
    for(size_t i = 0; i < output1.size(); ++i){
        if(std::abs(output1[i] - output2[i]) > tolerance){
            // stdout the difference 
            std::cout << "Difference at index " << i << " : " << std::abs(output1[i] - output2[i]) << " exceeds tolerance of " << tolerance << std::endl;
            return false; // Results are not valid within the specified tolerance
        }
    }
    return true; // All results are valid within the specified tolerance
}

/**
    @brief Helper function to write a vector of strings to a CSV file.
    @param path_to_file_with_filename The full path to the CSV file including the filename.
    @param content A vector of strings, each representing a line to be written to the CSV

    if csv is not present, create a new file.
*/
void utils::writeCSV(
    const std::string& path_to_file_with_filename, 
    const std::vector<std::string> &content,
    const std::vector<std::string> &headers
){
    std::ofstream file_to_write_to; 
    //if file does not exist, create it and write headers
    file_to_write_to.open(path_to_file_with_filename, std::ios::out | std::ios::app);
    if(!file_to_write_to.is_open()){
        throw std::runtime_error("Could not open file: " + path_to_file_with_filename);
    }
    
    int col_count = headers.size();
    // check if header is already present
    file_to_write_to.seekp(0, std::ios::end);
    if(file_to_write_to.tellp() != 0){
        // move pointer to next row 
        file_to_write_to << "\n";
    }
    else{
        for (size_t i = 0; i < headers.size(); ++i) {
            file_to_write_to << headers[i];
            if (i != headers.size() - 1) {
                file_to_write_to << ",";
            }
        }
        file_to_write_to << "\n";
    }

    // write content
    // file is not empty and skip first row 
    int inital_col = 0;
    for(const auto& line : content){
        if(inital_col >= col_count){
            file_to_write_to << "\n";
            inital_col = 0;
        }
        ++inital_col; // start at 1
        file_to_write_to << line << ","; 
    }
    
   
    std::cout << "Data written to CSV file: " << path_to_file_with_filename << std::endl;
    file_to_write_to.close();
}

void utils::checkcuComplexArray(cuComplex* data, int width, int height, const std::string& array_name){
    std::vector<cuComplex> host_data(width * height);
    cudaMemcpy(host_data.data(), data, width * height * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    std::cout << "Contents of " << array_name << " :" << std::endl;
    
    // check dims 
    std::cout << "Dimensions: " << height << " x " << width << std::endl; 
}

void utils::saveOutputImage(
    const std::string& path_to_file_with_filename,
    const std::vector<float>& output,
    int out_width,
    int out_height
){
    
    // create cv::Mat from output vector
    cv::Mat image_float(out_height, out_width, CV_32FC1, const_cast<float*>(output.data()));

    // Normalize to [0, 1.0] range
    cv::Mat normalised; 
    double minVal, maxVal;
    cv::minMaxLoc(image_float, &minVal, &maxVal);
    
    // avoid division by zero if image is solid color
    if (maxVal - minVal > 1e-5) {
        image_float.convertTo(normalised, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    } else {
        normalised = cv::Mat::zeros(out_height, out_width, CV_8UC1);
    }
    
    bool success = cv::imwrite(path_to_file_with_filename, normalised);
    if(!success){
        throw std::runtime_error("Failed to save image to: " + path_to_file_with_filename);
    } else {
        std::cout << "Image saved to: " << path_to_file_with_filename << std::endl;
    }
}


void utils::printConvResult(std::vector<float>& output, int out_width, int out_height){
    std::cout << "Convolution Result: " << std::endl;
    for(int i = 0; i < out_height; ++i){
        std::cout << "Row " << i << " : ";
        for(int j = 0; j < out_width; ++j){
            std::cout << std::setprecision(3) << "(" << j << ")" << output[i * out_width + j] << " ";
        }
        
        std::cout << std::endl;
    }
}


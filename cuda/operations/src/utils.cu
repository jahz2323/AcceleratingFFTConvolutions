#include "utils.cuh"



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
    @brief Helper function to write a vector of strings to a CSV file.
    @param path_to_file_with_filename The full path to the CSV file including the filename.
    @param content A vector of strings, each representing a line to be written to the CSV

    if csv is not present, create a new file.
*/
void utils::writeCSV(
    const std::string& path_to_file_with_filename, 
    const std::vector<std::string> &content
){
    std::ofstream file_to_write_to; 
    file_to_write_to.open(path_to_file_with_filename, std::ios::out | std::ios::app);
    if(!file_to_write_to.is_open()){
        throw std::runtime_error("Failed to open file: " + path_to_file_with_filename);
        return;
    }
    int col_count = 6;
    // check if header is already present
    file_to_write_to.seekp(0, std::ios::end);
    if(file_to_write_to.tellp() != 0){
        // move pointer to next row 
        file_to_write_to << "\n";
    }
    else{
        file_to_write_to << " Conv_Method , Input_Dimensions , Filter_Dimensions , Stride , Padding , Time_ms \n";
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
    delete[] host_data.data();
}

void utils::printConvResult(std::vector<float>& output, int out_width, int out_height){
    std::cout << "Convolution Result: " << std::endl;
    for(int i = 0; i < out_height; ++i){
        for(int j = 0; j < out_width; ++j){
            std::cout << output[i * out_width + j] << " ";
        }
        std::cout << std::endl;
    }
}


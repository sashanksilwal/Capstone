#include <string>
#include <string.h>
#include <sstream>
#include <stdint.h>
#include <assert.h>
#include <stdexcept>
#include <setjmp.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <unordered_set>
#include <fstream>

#include <onnxruntime_cxx_api.h>

using namespace std;
std::vector<std::string> classification_labels = {
    "marketing",
    "cdn",
    "tag-manager",
    "video",
    "customer-success",
    "utility",
    "ads",
    "analytics",
    "hosting",
    "content",
    "social",
    "other"
};

std::vector<std::string> get_scripts_features(std::string data, std::unordered_set<std::string> kws, std::vector<std::string> features) {
    std::vector<std::string> resultant_features;
    std::vector<std::string> scripts_kws;

    /* iterating through a list of keywords, removing spaces from each keyword, and then searching for that keyword in a given 
    string data. If a keyword is found in data, it is added to a list of keywords called scripts_kws. The code does this by iterating 
    through data one character at a time and checking for a match with the keyword. If a match is found, the keyword is added to scripts_kws.*/
    for (auto& kw : kws) {
        // remove spaces from kw
        std::string kw_no_spaces = kw;
        kw_no_spaces.erase(std::remove(kw_no_spaces.begin(), kw_no_spaces.end(), ' '), kw_no_spaces.end());
        std::string kw1 = "." + kw_no_spaces + "(";

        int count = 0;
        size_t pos = data.find(kw1);
        while (pos != std::string::npos) {
            count++;
            pos = data.find(kw1, pos + 1);
        }
         
        for (int i = 0; i < count; i++) {
            scripts_kws.push_back(kw_no_spaces);
        }
    }
    
    /*iterating through a list of features, checking if each feature contains the character "|". 
    If a feature does not contain the character "|", the code counts the number of times the feature appears
     in a list of keywords called scripts_kws and adds the feature to a list of resultant features the same 
     number of times. If a feature contains the character "|", the code splits the feature into individual 
     keywords using the character "|" as a delimiter and counts the number of times each individual keyword 
     appears in scripts_kws. If all individual keywords appear in scripts_kws, the original, unmodified feature 
     is added to the list of resultant features. This process is repeated for each feature in the list of features.*/
    for (auto& ft : features) {
    // remove space from ft
        ft.erase(std::remove(ft.begin(), ft.end(), ' '), ft.end());

        if (ft.find("|") == std::string::npos) {
            //count the number of times the feature appears in the script_kws vector
            int count = 0;
            for (auto& kw : scripts_kws) {
                if (kw == ft) {
                    count++;
                }
            }
            for (int i = 0; i < count; i++) {
                resultant_features.push_back(ft);
            }
        } else {
            std::vector<std::string> singular_kws;
            std::string delimiter = "|";
            size_t pos = 0;
            std::string token;

            std::string ft_copy = ft;

            while ((pos = ft.find(delimiter)) != std::string::npos) {
                token = ft.substr(0, pos);
                singular_kws.push_back(token);
                ft.erase(0, pos + delimiter.length());
            }
            singular_kws.push_back(ft);

            int count = 0;
            for (auto& kw : singular_kws) {
                if (std::count(scripts_kws.begin(), scripts_kws.end(), kw) > 0) {
                    count++;
                }
            }
            if (count == singular_kws.size()) {
                resultant_features.push_back(ft_copy);
            }
        }
    }
    return resultant_features;
}

/* function that takes a string data and two lists of keywords, classification_kws and classification_features, as inputs.
 The function first calls another function, get_scripts_features, which takes data, classification_kws, and classification_features 
 as inputs and returns a list of keywords. The function then iterates through this list of keywords, concatenating each 
 keyword with a space character, and returns the resulting string.*/

std::string get_scripts_classification_features(std::string data, std::unordered_set<std::string> classification_kws, std::vector<std::string> classification_features) {
    std::vector<std::string> features = get_scripts_features(data, classification_kws, classification_features);
    std::stringstream resultant_features;
    for (auto& ft : features) {
        resultant_features << ft << " ";
    }
    return resultant_features.str();
}

/*functioon code is using the ONNX Runtime library to load and run two machine learning models, sess_classification and sess_classification_tfidf. 
The code first calls the get_scripts_classification_features function to extract a list of keywords and features from a given script. 
The code then feeds the list of keywords and features to the models as input and runs the models to make a prediction. The models' 
outputs are then extracted and used to calculate the predicted class and confidence of the prediction.*/
std::pair<int, float> predict(std::string script, std::unordered_set<std::string> kws, std::vector<std::string> features) {


    std::string reduced_script = get_scripts_classification_features(script, kws, features);
    std::string reduced_script_copy = reduced_script;

    Ort::Env env;
    Ort::RunOptions runOptions;
    Ort::SessionOptions sessionOptions;
    auto modelPath = "/Users/sashanksilwal/Developer/Capstone/Phase1_CreatingAModel/models/classification_test.onnx";
    auto modelPath_tfidf = "/Users/sashanksilwal/Developer/Capstone/Phase1_CreatingAModel/models/classification_tfidf_test.onnx";

    // initialize session 
    Ort:: Session sess_classification = Ort::Session(env, modelPath, sessionOptions);
    Ort:: Session sess_classification_tfidf = Ort::Session(env, modelPath_tfidf, sessionOptions);

     
    //*************************************************************************
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    // cout << "sess_classification_tfidf " << endl;
    const size_t num_input_nodes_tfidf = sess_classification_tfidf.GetInputCount();
    std::vector<Ort::AllocatedStringPtr> input_names_ptr_tfidf;
    std::vector<const char*> input_node_names_tfidf;
    input_names_ptr_tfidf.reserve(num_input_nodes_tfidf);
    input_node_names_tfidf.reserve(num_input_nodes_tfidf);
    std::vector<int64_t> input_node_dims_tfidf;  
    std::vector<const char*> input_names_tfidf;
    std::vector<const char*>  output_node_names_tfidf;
    // iterate over all input nodes
    for (size_t i = 0; i < num_input_nodes_tfidf; i++) {
        // print input node names
        auto input_name = sess_classification_tfidf.GetInputNameAllocated(i, allocator);
        // std::cout << "Input " << i << " : name = " << input_name.get() << std::endl;
        input_node_names_tfidf.push_back(input_name.get());
        input_names_ptr_tfidf.push_back(std::move(input_name));

        // print input node types
        auto type_info = sess_classification_tfidf.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        // std::cout << "Input " << i << " : type = " << type << std::endl;

        // print input shapes/dims
        // std::cout << "Input " << i << " : num_dims = " << input_node_dims_tfidf.size() << '\n';
        // for (size_t j = 0; j < input_node_dims_tfidf.size(); j++) {
        //     std::cout << "Input " << i << " : dim[" << j << "] = " << input_node_dims_tfidf[j] << '\n';
        // }
    }
        auto input_num_tfidf = sess_classification_tfidf.GetInputCount();
        auto output_num_tfidf = sess_classification_tfidf.GetOutputCount();

        Ort::AllocatorWithDefaultOptions ort_alloc;
   
        for (size_t i = 0; i < output_num_tfidf; i++) {
            output_node_names_tfidf.push_back(sess_classification_tfidf.GetOutputNameAllocated(i, ort_alloc).release());
        }
        // print the input and output names
        // for (size_t i = 0; i < input_num_tfidf; i++) {
        //     std::cout << "Input Name " << i << ": " << input_node_names_tfidf[i] << std::endl;
        // }
        // for (size_t i = 0; i < output_num_tfidf; i++) {
        //     std::cout << "Output Name " << i << ": " << output_node_names_tfidf[i] << std::endl;
        // }
        // cout <<"*************************************************************************" <<endl;

        // std::cout << std::flush;

    
    //*************************************************************************
    Ort::AllocatorWithDefaultOptions allocator1;

    // print number of model input nodes
    // cout << "sess_classification" << endl;
    const size_t num_input_nodes = sess_classification.GetInputCount();
    std::vector<Ort::AllocatedStringPtr> input_names_ptr;
    std::vector<const char*> input_node_names;
    input_names_ptr.reserve(num_input_nodes);
    input_node_names.reserve(num_input_nodes);
    std::vector<int64_t> input_node_dims;  

    // iterate over all input nodes
    for (size_t i = 0; i < num_input_nodes; i++) {
        // print input node names
        auto input_name = sess_classification.GetInputNameAllocated(i, allocator1);
        // std::cout << "Input " << i << " : name = " << input_name.get() << std::endl;
        input_node_names.push_back(input_name.get());
        input_names_ptr.push_back(std::move(input_name));

        // print input node types
        auto type_info = sess_classification.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        // std::cout << "Input " << i << " : type = " << type << std::endl;

        // // print input shapes/dims
        // std::cout << "Input " << i << " : num_dims = " << input_node_dims.size() << '\n';
        // for (size_t j = 0; j < input_node_dims.size(); j++) {
        //     std::cout << "Input " << i << " : dim[" << j << "] = " << input_node_dims[j] << '\n';
        // }
      
    }
    // Code to test out the input and output nodes of tfidf model
    // std::vector<const char*> input_names;
    std::vector<const char*>  output_node_names;
    auto input_num = sess_classification.GetInputCount();
    auto output_num = sess_classification.GetOutputCount();

    for (size_t i = 0; i < output_num; i++) {
        output_node_names.push_back(sess_classification.GetOutputNameAllocated(i, ort_alloc).release());
    }
    // print the input and output names
    // for (size_t i = 0; i < input_num; i++) {
    //     std::cout << "Input Name " << i << ": " << input_node_names[i] << std::endl;
    // }
    // for (size_t i = 0; i < output_num; i++) {
    //     std::cout << "Output Name " << i << ": " << output_node_names[i] << std::endl;
    // }
    // cout <<"*************************************************************************" <<endl;
    // std::cout << std::flush;
    //*************************************************************************

    constexpr size_t input_tensor_size_tfidf = 1 * 499;
    std::vector<std::string> input_tensor_values_tfidf(input_tensor_size_tfidf);
    constexpr size_t input_tensor_size = 1 * 499;
    // std::vector<std::string> input_tensor_values(input_tensor_size);
    // std::vector<const char*> output_node_names = {"output_label", "output_probability"};

    // initialize input data with values  
    for (unsigned int i = 0; i < input_tensor_size_tfidf; i++) input_tensor_values_tfidf[i] = reduced_script;


    float* values_ret;
    int max = 0;
    try {
        // Running the inference
        input_node_dims_tfidf = {1, 1};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<std::string>(memory_info, input_tensor_values_tfidf.data(), input_tensor_size_tfidf, input_node_dims_tfidf.data(), 2);
        assert(input_tensor.IsTensor());
        auto output_tensors = sess_classification_tfidf.Run(Ort::RunOptions{nullptr}, input_node_names_tfidf.data(), &input_tensor, 1, output_node_names_tfidf.data(), 1);
        assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
        float* floatarr = output_tensors.front().GetTensorMutableData<float>();
        std::vector<float> input_tensor_values(499);

        for (int i = 0; i < 499; i++) {
            if (floatarr[i] != 0) {
                input_tensor_values[i] =  floatarr[i];
                // std::cout << "Score for class [" << i << "] =  " << floatarr[i] << '\n';
            }   
        }
        // std::cout << std::flush;
    
        // // run the classification model using the output of the tfidf model floatarr
        // OrtGetTensorShapeElementCount
        input_node_dims = {1, 499};
        // input_node_dims = {1,1};
        auto input_tensor2 = Ort::Value::CreateTensor<float>(memory_info, floatarr, input_tensor_size, input_node_dims.data(), 2);
        assert(input_tensor2.IsTensor());
        assert(input_tensor2.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        // print info of input tensor 2
        // std::cout << "Input 2 : num_dims = " << input_tensor2.GetTensorTypeAndShapeInfo().GetDimensionsCount() << '\n';
        // std::cout << "Input 2 : type = " << input_tensor2.GetTensorTypeAndShapeInfo().GetElementType() << '\n';
        
        auto output_tensors2 = sess_classification.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor2, 1, output_node_names.data(), 2);
        assert(output_tensors2.size() == 2 && output_tensors2.front().IsTensor());
        size_t num_values = output_tensors2[1].GetCount();
        const size_t N = 1;
        const int NUM_KV_PAIRS = 12;
         

        // for (size_t idx = 0; idx < 1; ++idx) {
        Ort::Value map_out = output_tensors2[1].GetValue(static_cast<int>(0), allocator);

        Ort::Value keys_ort = map_out.GetValue(0, allocator);
        int64_t* keys_ret = keys_ort.GetTensorMutableData<int64_t>();
        Ort::Value values_ort = map_out.GetValue(1, allocator);
        values_ret = values_ort.GetTensorMutableData<float>();
       
        for (int i = 0; i < 12; i++) {
            // std::cout << "Score for class [" << i << "] =  " << values_ret[i] << '\n';
            if (values_ret[i] > values_ret[max]) {
                max = i;  
            }
        }
     
    } catch (const Ort::Exception& exception) {
        std::cout << "ERROR running model inference: " << exception.what() << endl;
        exit(-1);
    }
   
    // return the index of the highest score and the score 
    return std::make_pair(max, values_ret[max]);
  
   
}


int main() {
    
    // read a file 
    // std::ifstream file1("/Users/sashanksilwal/Developer/Capstone/Phase1_CreatingAModel/JS/07faba79361a06b1fcbc3ce19a714e.m");
    // read all the files in a directory
    // open a file to write classification results
    std::ofstream outfile;
    outfile.open("/Users/sashanksilwal/Developer/Capstone/Phase2_ComparingOutputs_Benchmarking/c++_output3.csv");

    std::string path = "/Users/sashanksilwal/Developer/Capstone/Phase1_CreatingAModel/JS/";
    std::vector<std::string> files;
    double total_time;

    for (const auto & entry : std::filesystem::directory_iterator(path)) {
    
        std::ifstream file1( entry.path());
        std::string input((std::istreambuf_iterator<char>(file1)), std::istreambuf_iterator<char>());
        std::vector<std::string> input_array;
        std::unordered_set<std::string> classification_kws;

        // open the file and read the data
        std::ifstream file("/Users/sashanksilwal/Developer/Capstone/Phase2_ComparingOutputs_Benchmarking/classification_features.txt");
        std::string str;
        // skip 2 lines 
        // std::getline(file, str);
        // std::getline(file, str);
        while (std::getline(file, str))
        {
            // str.erase(std::remove(str.begin(), str.end(), '"'), str.end());
            // str.erase(std::remove(str.begin(), str.end(), ','), str.end());

            input_array.push_back(str);
        }
        
    
        // Iterate through each input string
        for (auto& element : input_array) {
            // Split the input string by the "|" character
            std::string feature = element;
            std::string delimiter = "|";
            size_t pos = 0;
            std::string token;

            while ((pos = feature.find(delimiter)) != std::string::npos) {
                // Extract the individual keyword from the input string
                token = feature.substr(0, pos);
                // Remove any spaces from the keyword
                token.erase(std::remove(token.begin(), token.end(), ' '), token.end());

                // Add the keyword to the classification_kws set
                classification_kws.insert(token);

                // Remove the keyword from the input string
                feature.erase(0, pos + delimiter.length());
            }

            // Remove any remaining spaces from the input string
            feature.erase(std::remove(feature.begin(), feature.end(), ' '), feature.end());

            // Add the remaining keyword to the classification_kws set
            classification_kws.insert(feature);
        }
      
       // Measure the time taken to run the inference function
        auto start = std::chrono::high_resolution_clock::now();

        // Call the inference function
        std::pair<int, float> index = predict(input, classification_kws, input_array);

        // Calculate the elapsed time
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        
        //get the file name
        std::string file_ = entry.path();
        std::string file_name = file_.substr(file_.find_last_of("/\\") + 1);
        std::cout << "file_name "<<file_name << std::endl;
        // cout << "[" << classification_labels[index.first] << "," << file_name<<","<< index.second <<"]"<< '\n';
        outfile   << file_name <<","<< classification_labels[index.first] << "," << index.second  << '\n';

        // total time taken for inference
        total_time += elapsed.count();
        
        
    }
   cout << "Total time taken for inference: " << total_time << " s\n";
    outfile.close();
    return 0;

}


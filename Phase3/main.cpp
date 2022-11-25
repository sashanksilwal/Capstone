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
#include <unordered_map>
#include <fstream>

#include <onnxruntime_cxx_api.h>

using namespace std;
std::unordered_map<int, std::string> classification_labels;

std::vector<std::string> get_scripts_features(std::string data, std::vector<std::string> kws, std::vector<std::string> features) {
    std::vector<std::string> resultant_features;
    std::vector<std::string> scripts_kws;

    for (auto& kw : kws) {
         // remove spaces from kw
        std::string kw_no_spaces = kw;
        kw_no_spaces.erase(std::remove(kw_no_spaces.begin(), kw_no_spaces.end(), ' '), kw_no_spaces.end());
        std::string kw1 = "." + kw_no_spaces + "(";

        int M = kw1.length();
        int N = data.length();
        int count = 0;

        for (int i = 0; i <= N - M; i++) {
        /* For current index i, check for
           pattern match */
        int j;
        for (j = 0; j < M; j++)
            if (data[i + j] != kw1[j])
                break;
 
        if (j == M) {
            count++;
        }

        }     
        for (int i = 0; i < count; i++) {
            scripts_kws.push_back(kw_no_spaces);
        }
    }
    
    for (auto& ft : features) {
        if (ft.find("|") == std::string::npos) {
            //count the number of times the feature appears in the script_kws vector
            int count =0 ;
            for (auto& kw : scripts_kws) {
                // remove space from kw
                kw.erase(std::remove(kw.begin(), kw.end(), ' '), kw.end());
                ft.erase(std::remove(ft.begin(), ft.end(), ' '), ft.end());
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
                token.erase(std::remove(token.begin(), token.end(), ' '), token.end());
                 
                singular_kws.push_back(token);
                ft.erase(0, pos + delimiter.length());
                ft.erase(std::remove(ft.begin(), ft.end(), ' '), ft.end());
                
            }
            
            singular_kws.push_back(ft);
             
            int count = 0;
            for (auto& kw : singular_kws) {
                if (std::count(scripts_kws.begin(), scripts_kws.end(), kw) > 0) {

                    count++;
                }
                 
            }
            if (count == singular_kws.size()) {
                // remove space from ft_copy
                ft_copy.erase(std::remove(ft_copy.begin(), ft_copy.end(), ' '), ft_copy.end());
                resultant_features.push_back(ft_copy);
            }
            
        }
    }
    return resultant_features;
}


string get_scripts_classification_features(std::string data, std::vector<std::string> classification_kws, std::vector<std::string> classification_features) {
    std::vector<std::string> features = get_scripts_features(data, classification_kws, classification_features);
    std::string resultant_features;
    for (auto& ft : features) {
        resultant_features += ft + " ";
     }
    return resultant_features;
}

std::string predict(std::string script, std::vector<std::string> kws, std::vector<std::string> features) {


    std::string reduced_script = get_scripts_classification_features(script, kws, features);
    std::string reduced_script_copy = reduced_script;

    Ort::Env env;
    Ort::RunOptions runOptions;
    Ort::SessionOptions sessionOptions;
    auto modelPath = "/Users/sashanksilwal/Developer/Capstone/Phase2_ComparingOutputs_Benchmarking/models/classification_test.onnx";
    auto modelPath_tfidf = "/Users/sashanksilwal/Developer/Capstone/Phase2_ComparingOutputs_Benchmarking/models/classification_tfidf_test.onnx";

    // initialize session 
    Ort:: Session sess_classification = Ort::Session(env, modelPath, sessionOptions);
    Ort:: Session sess_classification_tfidf = Ort::Session(env, modelPath_tfidf, sessionOptions);

    // create const char *const * input name float_input
    const char *const input_name_classification = "float_input";
    const char *const output_name_classification_tfidf = "variable";
    const char *const output_name_classification = "output_label";

    const char *const* input_name  = &input_name_classification;
    const char *const* output_name_tfidf = &output_name_classification_tfidf;
    const char *const* output_name = &output_name_classification;

    // std::vector<std::string> reduced_script_vector;
    // std::string delimiter = " ";
    // size_t pos = 0;
    // std::string token;
    // while ((pos = reduced_script.find(delimiter)) != std::string::npos) {
    //     token = reduced_script.substr(0, pos);
    //     reduced_script_vector.push_back(token);
    //     reduced_script.erase(0, pos + delimiter.length());
    // }
    //create a vector with one element from reduced_script
    std::vector<std::string> reduced_script_vector;
    reduced_script_vector.push_back(reduced_script);

    
    //*************************************************************************
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    cout << "sess_classification_tfidf " << endl;
    const size_t num_input_nodes_tfidf = sess_classification_tfidf.GetInputCount();
    std::vector<Ort::AllocatedStringPtr> input_names_ptr_tfidf;
    std::vector<const char*> input_node_names_tfidf;
    input_names_ptr_tfidf.reserve(num_input_nodes_tfidf);
    input_node_names_tfidf.reserve(num_input_nodes_tfidf);
    std::vector<int64_t> input_node_dims_tfidf;  

    // iterate over all input nodes
    for (size_t i = 0; i < num_input_nodes_tfidf; i++) {
        // print input node names
        auto input_name = sess_classification_tfidf.GetInputNameAllocated(i, allocator);
        std::cout << "Input " << i << " : name = " << input_name.get() << std::endl;
        input_node_names_tfidf.push_back(input_name.get());
        input_names_ptr_tfidf.push_back(std::move(input_name));

        // print input node types
        auto type_info = sess_classification.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::cout << "Input " << i << " : type = " << type << std::endl;

        // print input shapes/dims
        std::cout << "Input " << i << " : num_dims = " << input_node_dims_tfidf.size() << '\n';
        for (size_t j = 0; j < input_node_dims_tfidf.size(); j++) {
            std::cout << "Input " << i << " : dim[" << j << "] = " << input_node_dims_tfidf[j] << '\n';
        }
        cout <<"*************************************************************************" <<endl;
        std::cout << std::flush;

    }

    
    //*************************************************************************
    Ort::AllocatorWithDefaultOptions allocator1;

    // print number of model input nodes
    cout << "sess_classification" << endl;
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
        std::cout << "Input " << i << " : name = " << input_name.get() << std::endl;
        input_node_names.push_back(input_name.get());
        input_names_ptr.push_back(std::move(input_name));

        // print input node types
        auto type_info = sess_classification.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::cout << "Input " << i << " : type = " << type << std::endl;

        // print input shapes/dims
        std::cout << "Input " << i << " : num_dims = " << input_node_dims.size() << '\n';
        for (size_t j = 0; j < input_node_dims.size(); j++) {
            std::cout << "Input " << i << " : dim[" << j << "] = " << input_node_dims[j] << '\n';
        }
        cout <<"*************************************************************************" <<endl;
        std::cout << std::flush;
    }

    //*************************************************************************
    // Code to test out the input and output nodes of tfidf model
    std::vector<const char*> input_names;
    std::vector<const char*>  output_names;
    auto input_num = sess_classification.GetInputCount();
    auto output_num = sess_classification.GetOutputCount();

    Ort::AllocatorWithDefaultOptions ort_alloc;
    for (size_t i = 0; i < input_num; i++) {
        input_names.push_back(sess_classification.GetInputNameAllocated(i, ort_alloc).release());
    }
    for (size_t i = 0; i < output_num; i++) {
        output_names.push_back(sess_classification.GetOutputNameAllocated(i, ort_alloc).release());
    }
    // // print the input and output names
    for (size_t i = 0; i < input_num; i++) {
        std::cout << "Input Name " << i << ": " << input_names[i] << std::endl;
    }
    for (size_t i = 0; i < output_num; i++) {
        std::cout << "Output Name " << i << ": " << output_names[i] << std::endl;
    }
    //*************************************************************************

    constexpr size_t input_tensor_size = 1 * 499;
    std::vector<std::string> input_tensor_values(input_tensor_size);
    std::vector<const char*> output_node_names_tfidf = {"variable"};
    std::vector<const char*> output_node_names = {"output_label", "output_probability"};

    // initialize input data with values  
    for (unsigned int i = 0; i < input_tensor_size; i++) input_tensor_values[i] = reduced_script;

    
    //  // print number of model input nodes
    // const size_t num_input_nodes = sess_classification.GetInputCount();
    // std::vector<Ort::AllocatedStringPtr> input_names_ptr;
    // std::vector<const char*> input_node_names;
    // input_names_ptr.reserve(num_input_nodes);
    // input_node_names.reserve(num_input_nodes);

    // std::cout << "Number of inputs = " << num_input_nodes << std::endl;


    try {
      

        // Running the inference
        input_node_dims_tfidf = {1, 1};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<std::string>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims_tfidf.data(), 2);
        assert(input_tensor.IsTensor());
        auto output_tensors = sess_classification_tfidf.Run(Ort::RunOptions{nullptr}, input_node_names_tfidf.data(), &input_tensor, 1, output_node_names_tfidf.data(), 1);
        assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
        float* floatarr = output_tensors.front().GetTensorMutableData<float>();
        cout << &output_tensors[0] << endl;
        input_node_dims = { 1, 12};
        for (int i = 0; i < 499; i++) {
            std::cout << "Score for class [" << i << "] =  " << floatarr[i] << '\n';
        }
        // std::cout << std::flush;

        // run the classification model using the output of the tfidf model floatarr
        // auto prediction = sess_classification.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &output_tensors[0], 499, output_node_names.data(), 2);
        // // assert(prediction.size() == 1 && prediction.front().IsTensor());        // print tfidf_representation
        // // print prediction
        // // assert(abs(floatarr[0] - 0.000045) < 1e-6);
        //  float* pred = prediction.front().GetTensorMutableData<float>();
        
        // for (int i = 0; i < 15; i++) {
        //     std::cout << "Score for class [" << i << "] =  " << pred[i] << '\n';
        // }
     
    } catch (const Ort::Exception& exception) {
        cout << "ERROR running model inference: " << exception.what() << endl;
        exit(-1);
    }
   // return classification_labels[prediction[0]] 
    // std::vector<float> tfidf_representation = sess_classification_tfidf.Run(Ort::RunOptions{nullptr}, input_name_tfidf, reduced_script_vector, 1, nullptr, 1);
    return reduced_script;
   
}


int main() {
    
    // create a hashmap to store classification labels
    classification_labels[0] = "marketing";
    classification_labels[1] = "cdn";
    classification_labels[2] = "tag-manager";
    classification_labels[3] = "video";
    classification_labels[4] = "customer-success";
    classification_labels[5] = "utility";
    classification_labels[6] = "ads";
    classification_labels[7] = "analytics";
    classification_labels[8] = "hosting";
    classification_labels[9] = "content";
    classification_labels[10] = "social";
    classification_labels[11] = "other";

    // read a file 
    std::ifstream file1("/Users/sashanksilwal/Developer/Capstone/Phase1_CreatingAModel/JS/2cd462471abd91a76843fcc119efa5.m");
    // sore into a string input 
    std::string input((std::istreambuf_iterator<char>(file1)), std::istreambuf_iterator<char>());
    input = "document.getElementsByTagName() document.getElementsByTagName() a.contains() a.max() a.removeChild()";
    std::vector<std::string> input_array;
    // open the file and read the data
    std::ifstream file("/Users/sashanksilwal/Developer/Capstone/Phase2_ComparingOutputs_Benchmarking/classification_features.json");
    std::string str;
    // skip 2 lines 
    std::getline(file, str);
    std::getline(file, str);
    while (std::getline(file, str))
    {
        str.erase(std::remove(str.begin(), str.end(), '"'), str.end());
        str.erase(std::remove(str.begin(), str.end(), ','), str.end());

        input_array.push_back(str);
    }
    std::vector<std::string> classification_kws;
   
    for (auto& element : input_array) {
        // split the string by |
        std::string feature = element;
        std::string delimiter = "|";
        size_t pos = 0;
        std::string token;
        while ((pos = feature.find(delimiter)) != std::string::npos) {
            token = feature.substr(0, pos);
            token.erase(std::remove(token.begin(), token.end(), ' '), token.end());
            classification_kws.push_back(token);
            feature.erase(0, pos + delimiter.length());
            // remove the space from the feature
        }
        feature.erase(std::remove(feature.begin(), feature.end(), ' '), feature.end());
        classification_kws.push_back(feature);
    }
    // remove duplicates from the vector classification_kws
    std::sort(classification_kws.begin(), classification_kws.end());     
    auto last = std::unique(classification_kws.begin(), classification_kws.end());
    classification_kws.erase(last, classification_kws.end());

    std::string reduced_script = predict(input, classification_kws, input_array);
    std::cout << "Reduced " << reduced_script << std::endl;

    return 0;

}


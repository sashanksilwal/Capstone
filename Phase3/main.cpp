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

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx-executor");
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

    
    
    try {
        // Array of C style strings of length output_count that is the list of output names
        Ort::AllocatorWithDefaultOptions allocator;
        auto allocatorInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_node_dims_classification = {1, 1};  
        Ort::Value input_tensor_classification = Ort::Value::CreateTensor<std::string>(allocatorInfo,  &reduced_script  , reduced_script.size(), input_node_dims_classification.data(), input_node_dims_classification.size());
        std::vector<Ort::Value> output_tensor_classification = sess_classification_tfidf.Run(Ort::RunOptions{nullptr}, input_name, &input_tensor_classification, 1, output_name_tfidf, 499);
        // std::vector<Ort::Value> prediction = sess_classification.Run(Ort::RunOptions{nullptr}, input_name, &input_tensor_classification, 1, output_name, 12);
        
  
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


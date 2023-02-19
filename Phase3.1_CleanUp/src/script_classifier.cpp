#include "script_classifier.h"

using namespace std;




std::vector<std::string> ScriptClassifier::get_scripts_features(std::string data, std::unordered_set<std::string> kws, std::vector<std::string> features) {
    std::vector<std::string> resultant_features;
    std::vector<std::string> scripts_kws;

     /* iterating through a list of keywords, removing spaces from each keyword, and then searching for that keyword in a given 
    string data. If a keyword is found in data, it is added to a list of keywords called scripts_kws. The code does this by iterating 
    through data one character at a time and checking for a match with the keyword. If a match is found, the keyword is added to scripts_kws.*/
    
    for (auto& kw : kws) {
        // remove spaces from kw
        // Remove spaces from kw
        std::string kw_no_spaces = kw;
        kw_no_spaces.erase(std::remove(kw_no_spaces.begin(), kw_no_spaces.end(), ' '), kw_no_spaces.end());
        std::string kw1 = "." + kw_no_spaces + "(";

        // Find the first occurrence of kw1 in data
        size_t pos = data.find(kw1);

        // Count the number of occurrences of kw1 in data
        int count = 0;
        while (pos != std::string::npos) {
            count++;
            pos = data.find(kw1, pos + 1);
        }

        // Add kw to scripts_kws for each occurrence of kw1 in data
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
std::string ScriptClassifier::get_scripts_classification_features(std::string data, std::unordered_set<std::string> classification_kws, std::vector<std::string> classification_features) {
    std::vector<std::string> features = get_scripts_features(data, classification_kws, classification_features);
    std::stringstream resultant_features;
    for (auto& ft : features) {
        resultant_features << ft << " ";
    }
    return resultant_features.str();
}


/*function code is using the ONNX Runtime library to load and run two machine learning models, sess_classification and sess_classification_tfidf. 
The code first calls the get_scripts_classification_features function to extract a list of keywords and features from a given script. 
The code then feeds the list of keywords and features to the models as input and runs the models to make a prediction. The models' 
outputs are then extracted and used to calculate the predicted class and confidence of the prediction.*/
std::pair<int, float> ScriptClassifier::predict(std::string script, std::unordered_set<std::string> kws, std::vector<std::string> features) {


    std::string reduced_script = get_scripts_classification_features(script, kws, features);
    std::string reduced_script_copy = reduced_script;

    Ort::Env env;
    Ort::RunOptions runOptions;
    Ort::SessionOptions sessionOptions;
   
    //*************************************************************************
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

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

    }
        auto output_num_tfidf = sess_classification_tfidf.GetOutputCount();

        Ort::AllocatorWithDefaultOptions ort_alloc;
   
        for (size_t i = 0; i < output_num_tfidf; i++) {
            output_node_names_tfidf.push_back(sess_classification_tfidf.GetOutputNameAllocated(i, ort_alloc).release());
        }
       
    
    //*************************************************************************
    Ort::AllocatorWithDefaultOptions allocator1;

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

    
      
    }
    // Code to test out the input and output nodes of tfidf model
    // std::vector<const char*> input_names;
    std::vector<const char*>  output_node_names;
    // auto input_num = sess_classification.GetInputCount();
    auto output_num = sess_classification.GetOutputCount();

    for (size_t i = 0; i < output_num; i++) {
        output_node_names.push_back(sess_classification.GetOutputNameAllocated(i, ort_alloc).release());
    }
   
    constexpr size_t input_tensor_size_tfidf = 1 * 499;
    std::vector<std::string> input_tensor_values_tfidf(input_tensor_size_tfidf);
    constexpr size_t input_tensor_size = 1 * 499;
   
    // initialize input data with values  
    for (unsigned int i = 0; i < input_tensor_size_tfidf; i++) input_tensor_values_tfidf[i] = reduced_script;


    float* values_ret;
    int max = 0;
    float max_val = 0;
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
        
        input_node_dims = {1, 499};
        // input_node_dims = {1,1};
        auto input_tensor2 = Ort::Value::CreateTensor<float>(memory_info, floatarr, input_tensor_size, input_node_dims.data(), 2);
        assert(input_tensor2.IsTensor());
        assert(input_tensor2.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      
        auto output_tensors2 = sess_classification.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor2, 1, output_node_names.data(), 2);
        assert(output_tensors2.size() == 2 && output_tensors2.front().IsTensor());

        // for (size_t idx = 0; idx < 1; ++idx) {
        Ort::Value map_out = output_tensors2[1].GetValue(static_cast<int>(0), allocator);

        Ort::Value keys_ort = map_out.GetValue(0, allocator);
        // int64_t* keys_ret = keys_ort.GetTensorMutableData<int64_t>();
        Ort::Value values_ort = map_out.GetValue(1, allocator);
        values_ret = values_ort.GetTensorMutableData<float>();
        max_val = values_ret[0];
        for (int i = 0; i < 12; i++) {
            
            if (values_ret[i] > values_ret[max]) {
                // std::cout << "Score for class [" << i << "] =  " << values_ret[i] << '\n';
                max = i;  
                max_val = values_ret[i];
            }
        }
     
    } catch (const Ort::Exception& exception) {
        std::cout << "ERROR running model inference: " << exception.what() << endl;
        exit(-1);
    }
   
    // return the index of the highest score and the score 
    return std::make_pair(max, max_val);
  
   
}
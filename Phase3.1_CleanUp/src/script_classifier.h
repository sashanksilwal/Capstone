#ifndef SCRIPT_CLASSIFIER_H
#define SCRIPT_CLASSIFIER_H

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


// The ScriptClassifier class is used to classify scripts into predefined categories.
class ScriptClassifier {
public:
    // Predicts the category of a script, given a set of keywords and features.
    std::pair<int, float> predict(std::string script, std::unordered_set<std::string> kws, std::vector<std::string> features);

    // A vector containing the names of the categories that scripts can be classified into.
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

    // Default constructor. Initializes the environment and the sessions for the classification models.
    ScriptClassifier() :
        env(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test")),
        sess_classification(env, "./classification.onnx", Ort::SessionOptions{nullptr}),
        sess_classification_tfidf(env, "./classification_tfidf.onnx", Ort::SessionOptions{nullptr})
    {
    }
 
    
private:
    // The ORT environment used to create the sessions for the classification models.
    Ort::Env env;
    // The ORT session used to classify scripts using the classification model.
    Ort::Session sess_classification;
    // The ORT session used to classify scripts using the classification-TFIDF model.
    Ort::Session sess_classification_tfidf;
    // A set of keywords used to extract features from scripts for classification.
    std::unordered_set<std::string> classification_kws;
    // A vector of features used to classify scripts.
    std::vector<std::string> classification_features;
    
    // Extracts features from a script, given a set of keywords and features.
    std::vector<std::string> get_scripts_features(std::string data, std::unordered_set<std::string> kws, std::vector<std::string> features);
    // Extracts classification features from a script, given a set of keywords and features.
    std::string get_scripts_classification_features(std::string data, std::unordered_set<std::string> classification_kws, std::vector<std::string> classification_features);

};

#endif
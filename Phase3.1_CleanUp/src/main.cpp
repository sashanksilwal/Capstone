#include <iostream>
#include <vector>
#include <unordered_set>
#include "helper.h"
#include "script_classifier.h"

int main() {


    std::ofstream outfile;
    outfile.open("../c++_output.csv");

    std::string path = "/Users/sashanksilwal/Developer/Capstone/Phase1_CreatingAModel/JS/";
    std::vector<std::string> files;
    double total_time = 0.0;

    std::ofstream timefile("timeit.csv");

    for (const auto & entry : std::__fs::filesystem::directory_iterator(path)) {
    
        std::ifstream file1( entry.path());
        std::string input((std::istreambuf_iterator<char>(file1)), std::istreambuf_iterator<char>());

     
        std::unordered_set<std::string> classification_kws;
        std::vector<std::string> input_array;

        readClassificationFeatures(classification_kws, input_array);

        // Measure the time taken to run the inference function
        auto start = std::chrono::high_resolution_clock::now();

        ScriptClassifier classifier;
        std::pair<int, float> index = classifier.predict(input, classification_kws, input_array);

        // Calculate the elapsed time
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        
        //get the file name
        std::string file_ = entry.path();
        std::string file_name = file_.substr(file_.find_last_of("/\\") + 1);
     
        outfile   << file_name <<","<< classifier.classification_labels[index.first] << "," << index.second  << '\n';

        // total time taken for inference
        timefile <<  file_name <<","<<   elapsed.count()<< '\n';
        total_time += elapsed.count();

    }
    std::cout << "Total time taken for inference: " << total_time << std::endl;
    outfile.close();

    return 0;
}

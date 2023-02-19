#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <algorithm>

/**
 * Reads classification features from a file and populates an unordered set of strings with these features.
 *
 * @param classification_kws The unordered set to be populated with the classification keywords.
 * @param input_array The vector to hold the input strings read from the file.
 */
void readClassificationFeatures( std::unordered_set<std::string>& classification_kws, std::vector<std::string>& input_array) {
    // open the file and read the data
    // set the filename
    std::string filename = "./classification_features.txt";
    std::ifstream file(filename);
    std::string str;

    while (std::getline(file, str))
    {
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
}
/**
 * @file file_reader.h
 * @brief This header file provides the function signature for a function to read classification features from a file.
 */

#ifndef FILE_READER_H
#define FILE_READER_H

#include <string>
#include <vector>

/**
 * @brief Reads classification features from a file and populates an unordered set and a vector.
 * 
 * This function takes in a reference to an unordered set of strings and a vector of strings.
 * It reads from a file containing the classification features and populates the set and the vector.
 * 
 * @param classification_kws A reference to an unordered set of strings to be populated with the classification features.
 * @param input_array A vector of strings to be populated with the input features.
 */
void readClassificationFeatures(std::unordered_set<std::string>& classification_kws, std::vector<std::string>& input_array);

#endif
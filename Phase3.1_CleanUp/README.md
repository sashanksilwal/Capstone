### Description of the files in the `src` directory

`script_classifier.h` and `script_classifier.cpp` contain the code for the ScriptClassifier class. This class has the following functions:

- `get_scripts_classification_features()`: This function takes a script and a set of keywords as input. It then extracts a list of features from the script.

- `get_scripts_features()`: This function takes a script as input. It then extracts a list of features from the script.

- `predict()`: This function takes a script, a set of keywords, and a list of features as input. It first calls the `get_scripts_classification_features()` function to extract a list of keywords and features from the given script. The function then feeds the list of keywords and features to the models as input and runs the models to make a prediction. The models' outputs are then extracted and used to calculate the predicted class and confidence of the prediction.

This code loads and runs two machine learning models using the ONNX Runtime library. The first model is used to classify scripts, while the second is used to apply TF-IDF (term frequency-inverse document frequency) weighting to the scripts.

`helper.h` and `helper.cpp` contain the code for the Helper class. This class has the following functions:

` readClassificationFeatures()`: This function reads the classification features from the classification_features.txt file.

### Running the code

To run the code, use the follwoing command:

make
./script_classifier


import pickle
import numpy as np

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from skl2onnx.common.data_types import DoubleTensorType, Int64TensorType
from sklearn.feature_extraction.text import TfidfVectorizer

# main function takes the file name to convert to onnx
def main(filename, num_features=1000, input_name='float_input', output_name='output', file_path_classification="./models"):
   
    classification_model = pickle.load(open('classification_model', 'rb'))   

    initial_type = [(input_name, FloatTensorType([   None, num_features]))]
    # onx = convert_sklearn(classification_model,   initial_types=initial_type,   options={"zipmap": False})
    onx = convert_sklearn(classification_model,   initial_types=initial_type,   options={id(1):{'raw_scores': True, "zipmap": False}})


    with open(file_path_classification, "wb") as f:
        f.write(onx.SerializeToString())
  
#  take the file name as input from the command line
if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

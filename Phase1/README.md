# Phase 1

## Introduction

Converting a machine learning model (sklearn classification and clustering) into C++ adaptable format (our choice is ONNX)

## How to run

### Prerequisites

- Python 3.6 or higher
- scikit-learn 0.20 or higher
- skl2onnx 1.5.0 or higher

### Installation

```bash
pip install -r requirements.txt
```

### Run

```bash
python convert2onnx.py <filename> <num_features> <input_name> <output_name> <path_output>
```

- filename: the name of the model file
- input_name: the name of the input variable
- output_name: the name of the output variable
- path_output: the path to save the converted model

### Example

```bash
python convert2onnx.py model.pkl 1000 float_input output ./model
```

# Phase 1

## Introduction

Converting a machine learning model (sklearn classification and clustering) into C++ adaptable format (our choice is ONNX)

## How to run

## Option 1:

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

## Option 2: Using Jupyter Notebook

### Prerequisites

- Python 3.6 or higher
- scikit-learn 0.20 or higher
- skl2onnx 1.5.0 or higher
- jupyter notebook

### Installation

```bash
pip install notebook
pip install -r requirements.txt
```

### Run

```bash
jupyter notebook
```

Change the name of the model file and parameters in the notebook and run the cells.

## References

- [ONNX](https://onnx.ai/)
- [skl2onnx](https://onnx.ai/sklearn-onnx/)
- [sklearn](https://scikit-learn.org/stable/)
- [jupyter notebook](https://jupyter.org/)
- [jupyter notebook tutorial](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)
- [jupyter notebook installation](https://jupyter.org/install)

## Multi-Scale Convolutional Neural Networks (CNNs) for Landslide Inventory Mapping from Remote Sensing Imagery and Landslide Susceptibility Mapping (LSM)



## Description

This paper proposes a method for Landslide Inventory Mapping and Landslide Susceptibility Mapping.

## Requirements

- tensorflow-gpu==2.6.0
- keras==2.6.0
- scikit-learn==0.23.2
- numpy
- pandas
- Keras Tuner
- GDAL
- h5py
- hyperopt

## Dataset

*The dataset is formatted as a number of csv files, where each column contains a different landslide impact factor and the last column also contains the labelled values.*

## Usage

1.for Semantic segmentation for landslide recognition
```
Training 
    python train.py

Inference
    python test.py
```

2.for Convolutional neural network landslide susceptibility evaluation
```
Training and inference
    python multiscale_3DCNN.py
```

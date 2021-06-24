Copyright (C) 2021 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file for details.

Authors: Thorir Mar Ingolfsson, Xiaying Wang, Michael Hersche, Alessio Burrello, Lukas Cavigelli, Luca Benini

# EEG-TCN
This project provides the experimental environment used to produce the results reported in the paper *ECG-TCN: Wearable Cardiac Arrhythmia Detection with a Temporal Convolutional Network* available on [arXiv](https://arxiv.org/abs/2103.13740). If you find this work useful in your research, please cite
```
@INPROCEEDINGS{9458520,
  author={Ingolfsson, Thorir Mar and Hersche, Xiaying Wang Michael and Burrello, Alessio and Cavigelli, Lukas and Benini, Luca},
  booktitle={2021 IEEE 3rd International Conference on Artificial Intelligence Circuits and Systems (AICAS)}, 
  title={ECG-TCN: Wearable Cardiac Arrhythmia Detection with a Temporal Convolutional Network}, 
  year={2021},
  volume={},
  number={},
  pages={1-4},
  doi={10.1109/AICAS51828.2021.9458520}}

```

## Getting started

### Prerequisites
* We developed and used the code behind ECG-TCN on [Ubuntu 18.04.3 LTS (Bionic Beaver) (64bit)](http://old-releases.ubuntu.com/releases/18.04.3/).
* The code behind ECG-TCN is based on Python3, and [Anaconda3](https://www.anaconda.com/distribution/) is required.
* We used [NVidia GTX1080 Ti GPUs](https://developer.nvidia.com/cuda-gpus) to accelerate the training of our models (driver version [396.44](https://www.nvidia.com/Download/driverResults.aspx/136950/en-us)). In this case, CUDA and the cuDNN library are needed (we used [CUDA 10.1](https://developer.nvidia.com/cuda-toolkit-archive)).

Also the dataset ECG5000 needs to be downloaded and put into the `/data` folder. It is available on [here](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)

### Installing
Navigate to ECG-TCN's main folder and create the environment using Anaconda:
```
$ conda env create -f ECG-TCN.yml -n ECG-TCN
```


## Usage
We provide the code to quantize the ECG-TCN model in the file `nemo_quantization.py` there we train, quantize and deploy the network to be used by DORY. We also provide code to quantize the same model with Tensorflow in the file `tensorflow_quantization.py` there we train, quantize and deploy the network to be used either straight on an MCU using TFlite as a library or using X-CUBE-AI.

Under `/utils` you find the data loading and model making files. Please note that because of the stochastic nature of training with GPUs it's very hard to fix every random variable in the backend. Therefore to reproduce the same or similar models one might need to train a couple of times in order to get the same highly accurate models we present.
### License and Attribution
Please refer to the LICENSE file for the licensing of our code.

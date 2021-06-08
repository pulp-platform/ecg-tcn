Copyright (C) 2021 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file for details.

Authors: Thorir Mar Ingolfsson, Xiaying Wang, Michael Hersche, Alessio Burello, Lukas Cavigelli, Luca Benini

# EEG-TCNet
This project provides the experimental environment used to produce the results reported in the paper *ECG-TCN: Wearable Cardiac Arrhythmia Detection with a Temporal Convolutional Network* available on [arXiv](https://arxiv.org/abs/2103.13740). If you find this work useful in your research, please cite
```
@misc{ingolfsson2021ecgtcn,
      title={ECG-TCN: Wearable Cardiac Arrhythmia Detection with a Temporal Convolutional Network}, 
      author={Thorir Mar Ingolfsson and  Xiaying Wang and Michael Hersche and Alessio Burello and Lukas Cavigelli and Luca Benini},
      booktitle={2021 3nd IEEE International Conference on Artificial Intelligence Circuits and Systems (AICAS)}, 
      year={2021},
      volume={},
      number={}
}

```

## Getting started

### Prerequisites
* We developed and used the code behind ECG-TCN on [Ubuntu 18.04.3 LTS (Bionic Beaver) (64bit)](http://old-releases.ubuntu.com/releases/18.04.3/).
* The code behind ECG-TCN is based on Python3, and [Anaconda3](https://www.anaconda.com/distribution/) is required.
* We used [NVidia GTX1080 Ti GPUs](https://developer.nvidia.com/cuda-gpus) to accelerate the training of our models (driver version [396.44](https://www.nvidia.com/Download/driverResults.aspx/136950/en-us)). In this case, CUDA and the cuDNN library are needed (we used [CUDA 10.1](https://developer.nvidia.com/cuda-toolkit-archive)).

### Installing
Navigate to ECG-TCN's main folder and create the environment using Anaconda:
```
$ conda env create -f ECG-TCN.yml -n ECG-TCN
```


## Usage


### License and Attribution
Please refer to the LICENSE file for the licensing of our code.

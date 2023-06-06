# MindFace
<div align="center">

English | [简体中文](README_CN.md)

</div>

| [Introduction](#introduction) | [Installation](#installation) | [Get Started](#get-started) | [Tutorials](#tutorials) | [Model List](#model-list) | [Notes](#notes) |

## Introduction

MindFace mainly has the following features.
- Unified Application Programming Interface

    MindFace provides a unified application programming interface for face recognition and detection by decoupling the models, so that the model can be called directly using the MindFace APIs, which greatly improves the ease of building algorithms for users

- Strong extensibility

    MindFace currently supports face recognition and detection, based on the unified APIs. MindFace is highly scalable, it can support lots of backbones, datasets, and loss functions. What's more, MindFace also supports many platforms, including CPU/GPU/Ascend.

</details>

### Benchmark Results

#### Recognition

The MindSpore implementation of ArcFace and has achieved great performance. We implemented three versions based on ResNet and MobileNet to meet different needs. Detailed results are shown in the table below.

| Datasets       | Backbone            | lfw         | cfp_fp      | agedb_30    | calfw | cplfw |
|:---------------|:--------------------|:------------|:------------|:------------|:------------|:------------|
| CASIA         | mobilefacenet-0.45g | 0.98483+-0.00425 | 0.86843+-0.01838 | 0.90133+-0.02118 | 0.90917+-0.01294 | 0.81217+-0.02232 |
| CASIA         | r50 | 0.98667+-0.00435 | 0.90357+-0.01300 | 0.91750+-0.02277 | 0.92033+-0.01122 | 0.83667+-0.01719 |
| CASIA         | r100 | 0.98950+-0.00366 | 0.90943+-0.01300 | 0.91833+-0.01655 | 0.92433+-0.01017 | 0.84967+-0.01904 |
| MS1MV2         | mobilefacenet-0.45g| 0.98700+-0.00364 | 0.88214+-0.01493 | 0.90950+-0.02076 | 0.91750+-0.01088 | 0.82633+-0.02014 |
| MS1MV2         | r50 | 0.99767+-0.00260 | 0.97186+-0.00652 | 0.97783+-0.00869 | 0.96067+-0.01121 | 0.92033+-0.01732 |
| MS1MV2         | r100 | 0.99383+-0.00334 | 0.96800+-0.01042 | 0.93767+-0.01724 | 0.93267+-0.01327 | 0.89150+-0.01763 |

#### Detection

For face detection, we choose resnet50 and mobilenet0.25 as the backbone, retinaface as the model architecture to achieve efficient performance of face detection. Detailed results are shown in the table below.

| Dataset | Backbone | Easy | Middle | Hard |
|:-|:-|:-:|:-:|:-:|
| WiderFace | mobileNet0.25 | 91.60% | 89.50% | 82.39% |
| WiderFace | ResNet50 | 95.81% | 94.89% | 90.10% |


## Installation

### Installing MindSpore in GPU

#### (1) Automatic Installation

Before using the automatic installation script, you need to make sure that the NVIDIA GPU driver is correctly installed on the system. The minimum required GPU driver version of CUDA 10.1 is 418.39. The minimum required GPU driver version of CUDA 11.1 is 450.80.02.

Run the following command to obtain and run the automatic installation script. The automatic installation script only supports the installation of MindSpore>=1.6.0.

```shell
wget https://gitee.com/mindspore/mindspore/raw/r1.8/scripts/install/ubuntu-gpu-pip.sh
# install MindSpore 1.8.1, Python 3.7 and CUDA 11.1
MINDSPORE_VERSION=1.8.1 bash -i ./ubuntu-gpu-pip.sh
# to specify Python and MindSpore version, taking Python 3.9 and MindSpore 1.6.0 as examples, use the following manners
# PYTHON_VERSION=3.9 CUDA_VERSION=10.1 MINDSPORE_VERSION=1.6.0 bash -i ./ubuntu-gpu-pip.sh
```

This script performs the following operations:

- Change the software source configuration to a HUAWEI CLOUD source.
- Install the dependencies required by MindSpore, such as GCC and gmp.
- Install Python3 and pip3 via APT and set them as default.
- Download and install CUDA and cuDNN.
- Install MindSpore GPU by pip.
- Install Open MPI if OPENMPI is set to `on`.

#### (2) Manual Installation

If some dependencies, such as CUDA, Python and GCC, have been installed in your system, it is recommended to install manually. First, refer to [Version List](https://www.mindspore.cn/versions) to select the version of MindSpore you want to install, and perform SHA-256 integrity check. Taking version 1.8.1 as an example, execute the following commands.

```shell
export MS_VERSION=1.8.1
```

Then install the latest version of MindSpore according to the CUDA version and Python version by following the following command.

```shell
# CUDA10.1 + Python3.7
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/gpu/x86_64/cuda-10.1/mindspore_gpu-${MS_VERSION/-/}-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# CUDA10.1 + Python3.8
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/gpu/x86_64/cuda-10.1/mindspore_gpu-${MS_VERSION/-/}-cp38-cp38-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# CUDA10.1 + Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/gpu/x86_64/cuda-10.1/mindspore_gpu-${MS_VERSION/-/}-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# CUDA11.1 + Python3.7
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/gpu/x86_64/cuda-11.1/mindspore_gpu-${MS_VERSION/-/}-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# CUDA11.1 + Python3.8
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/gpu/x86_64/cuda-11.1/mindspore_gpu-${MS_VERSION/-/}-cp38-cp38-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# CUDA11.1 + Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/gpu/x86_64/cuda-11.1/mindspore_gpu-${MS_VERSION/-/}-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

When the network is connected, dependency items are automatically downloaded during MindSpore installation (For details about the dependency, see required_package in `setup.py`.) .

When running models, you need to install additional dependencies based on `requirements.txt`.

### Installing MindSpore in Ascend 910

#### (1) Automatic Installation

Before running the automatic installation script, you need to make sure that the [Ascend AI processor software package](https://gitee.com/mindspore/docs/blob/r1.8/install/mindspore_ascend_install_pip.md#%E5%AE%89%E8%A3%85%E6%98%87%E8%85%BEai%E5%A4%84%E7%90%86%E5%99%A8%E9%85%8D%E5%A5%97%E8%BD%AF%E4%BB%B6%E5%8C%85) is correctly installed on your system. Run the following command to obtain and run the automatic installation script. The automatic installation script only supports the installation of MindSpore>=1.6.0.

```shell
wget https://gitee.com/mindspore/mindspore/raw/r1.8/scripts/install/euleros-ascend-pip.sh
# install MindSpore 1.8.1 and Python 3.7
# the default value of LOCAL_ASCEND is /usr/local/Ascend
MINDSPORE_VERSION=1.8.1 bash -i ./euleros-ascend-pip.sh
# to specify Python and MindSpore version, taking Python 3.9 and MindSpore 1.6.0 as examples
# and set LOCAL_ASCEND to /home/xxx/Ascend, use the following manners
# LOCAL_ASCEND=/home/xxx/Ascend PYTHON_VERSION=3.9 MINDSPORE_VERSION=1.6.0 bash -i ./euleros-ascend-pip.sh
```

This script performs the following operations:

- Install the dependencies required by MindSpore, such as GCC and gmp.
- Install Python3 and pip3 and set them as default.
- Install MindSpore Ascend by pip.
- Install Open MPI if OPENMPI is set to `on`.

The automatic installation script creates a virtual environment named `mindspore_pyXX` for MindSpore. Where `XX` is the Python version, such as Python 3.7, the virtual environment name is `mindspore_py37`. Run the following command to show all virtual environments.

```shell
conda env list
```

To activate the virtual environment, take Python 3.7 as an example, execute the following command.

```shell
conda activate mindspore_py37
```

#### （2）Manual Installation

If your system is one of Ubuntu 18.04/CentOS 7.6/OpenEuler 20.03/KylinV10 SP1, or some dependencies, such as Python and GCC, have been installed in your system, it is recommended to install manually. First, refer to [Version List](https://www.mindspore.cn/versions) to select the version of MindSpore you want to install, and perform SHA-256 integrity check. Taking version 1.8.1 as an example, execute the following commands.

```shell
export MS_VERSION=1.8.1
```

Then run the following commands to install MindSpore according to the system architecture and Python version.

```shell
# x86_64 + Python3.7
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/ascend/x86_64/mindspore_ascend-${MS_VERSION/-/}-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# x86_64 + Python3.8
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/ascend/x86_64/mindspore_ascend-${MS_VERSION/-/}-cp38-cp38-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# x86_64 + Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/ascend/x86_64/mindspore_ascend-${MS_VERSION/-/}-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64 + Python3.7
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/ascend/aarch64/mindspore_ascend-${MS_VERSION/-/}-cp37-cp37m-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64 + Python3.8
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/ascend/aarch64/mindspore_ascend-${MS_VERSION/-/}-cp38-cp38-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64 + Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/ascend/aarch64/mindspore_ascend-${MS_VERSION/-/}-cp39-cp39-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

When the network is connected, dependencies of MindSpore are automatically downloaded during the .whl package installation (For details about the dependency, see required_package in `setup.py` ).

When running models, you need to install additional dependencies based on `requirements.txt` .

## Get Started

- [Recognition get started](tutorials/recognition/get_started.md)
- [Detection get started]()

## Tutorials

We provide [tutorials](tutorials) for the recognition and detection task.

### Recognition

- [Get started](tutorials/recognition/get_started.md)
- [Learn about recognition configs](tutorials/recognition/config.md) 
- [Learn to reproduce the eval result and inference with a pretrained model](tutorials/recognition/inference.md) 
- [Learn about how to create dataset](tutorials/recognition/dataset.md)
- [Learn about how to train/finetune a pretrained model](tutorials/recognition/finetune.md)
- [Learn about how to use the loss function](tutorials/recognition/loss.md)
- [Learn about how to create model and custom model](tutorials/recognition/model.md)

### Detection

- [Learn about detection configs](tutorials/detection/config.md)  
- [Inference with a pretrained detection model](tutorials/detection/infer.md) 
- [Finetune a pretrained detection model on WiderFace](tutorials/detection/finetune.md)

## Supported Models

Currently, MindFace supports the following models. More models with pre-trained weights are under development and will be released in the near future.

<details>
<summary>Supported Models</summary>

- Detection
  - Resnet50
  - Mobilenet0.25
- Recognition
  - arcface-mobilefacenet-0.45g
  - arcface-r50
  - arcface-r100
  - arcface-vit-t
  - arcface-vit-s
  - arcface-vit-b
  - arcface-vit-l

</details>

Please click [here](mindface/detection/configs) to learn more about the detection model，and click [here](mindface/recognition/configs) to learn more about the recognition model.

## License

This project is released under the [Apache License 2.0](LICENSE.md).

## Feedbacks and Contact

The dynamic version is still under development, if you find any issue or have an idea on new features, please don't hesitate to contact us via [issue](https://github.com/mindlab-ai/mindface/issues).

## Acknowledgement

MindSpore is an open source project that welcome any contribution and feedback. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible as well as standardized toolkit to reimplement existing methods and develop their own new computer vision methods.

If you find *MindFace* useful in your research, please consider citing the following related papers:

```
@misc{MindFace 2022,
    title={{mindface}:mindface for face recognition and detection},
    author={mindface},
    howpublished = {\url{https://github.com/mindspore-lab/mindface/}},
    year={2022}
}

```

## Notes

* We have created our official repo about face research based on MindSpore.
* MindFace supports recognition and detection task.

## Contributing

*MindFace* is mainly maintained by the Cross-Media Intelligent Computing (**CMIC**) Laboratory, University of Science and Technology of China (**USTC**), and cooperated with Huawei Technologies Co., Ltd. 

The research topics of CMIC include multimedia computing, multi-modal information perception, cognition and synthesis. 

CMIC has published more than 200 journal articles and conference papers, including TPAMI, TIP, TMM, TASLP, TCSVT, TCYB, TITS, TOMM, TCDS, NeurIPS, ACL, CVPR, ICCV, MM, ICLR, SIGGRAPH, VR, AAAI, IJCAI. 

CMIC has received 6 best paper awards from premier conferences, including CVPR MAVOC, ICCV MFR, ICME, FG. 

CMIC has won 24 Grand Challenge Champion Awards from premier conferences, including CVPR, ICCV, MM, ECCV, AAAI, ICME.

**Main contributors:**

- [Jun Yu](https://github.com/harryjun-ustc), ``harryjun[at]ustc.edu.cn``
- Guochen xie, ``xiegc[at]mail.ustc.edu.cn``
- Shenshen Du, ``dushens[at]mail.ustc.edu.cn``
- Zhongpeng Cai, ``czp_2402242823[at]mail.ustc.edu.cn``
- Peng He, ``hp0618[at]mail.ustc.edu.cn``
- Liwen Zhang, ``zlw1113[at]mail.ustc.edu.cn``
- Hao Chang, ``changhaoustc[at]mail.ustc.edu.cn``
- Mohan Jing, ``jing_mohan@mail.ustc.edu.cn``
- Haoxiang Shi, ``shihaoxiang@mail.ustc.edu.cn``
- Keda Lu, ``lukeda@mail.ustc.edu.cn``
- Pengwei Li, ``lipw@mail.ustc.edu.cn``

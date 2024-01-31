# MCIPS-CL

<! --This repository contains the `PyTorch` implementation of the paper in the 2023 International Conference on Neural Information Processing (***[ICONIP 2023](http://www.iconip2023.org/) , Vol. 12***): 

**[A deep joint model of Multi-Slot Intent interaction with Second-Order Gate for Spoken Language Understanding](https://link.springer.com/chapter/10.1007/978-981-99-8148-9_4)**.

[Qingpeng Wen](mailto:wqp@mail2.gdut.edu.cn), [Bi Zeng](mailto:zb9215@gdut.edu.cn), [Pengfei Wei](mailto:wpf@gdut.edu.cn), [Huiting Hu](mailto:huhuiting@zhku.edu.cn)

In the following, we will guide you how to use this repository step by step.-->

## Architecture

<img src="overall.png">



Our code is based on Python 3.8.18 and PyTorch 2.1.0. Requirements are listed as follows:
> - torch==2.1.0
> - transformers==4.26.0
> - numpy==1.23.5
> - tqdm==4.66.1
> - seqeval==1.2.2

We highly suggest you using [Anaconda](https://www.anaconda.com) to manage your python environment.

## How to Run it

### Quick start
The script **run.py** in "example/ner/multimodal" acts as a main function to the project, you can run the experiments by replacing the unspecified options in the following command with the corresponding values:

```shell
    CUDA_VISIBLE_DEVICES=$1 python run.py -dd ${dataDir} -sd ${saveDir}
```

or run the script **run.py** directly via pycharm.

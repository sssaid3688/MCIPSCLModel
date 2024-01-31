# MCIPS-CL

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

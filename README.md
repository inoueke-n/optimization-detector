# Optimization Detector
This is the companion code for the paper 
> **Identifying Compiler and Optimization Options from Binary Code using Deep
> Learning Approaches**
>
> D. Pizzolotto, K. Inoue
>
> *Presented at the 36th IEEE International Conference on Software
> Maintenance and Evolution,
> [ICSME 2020](https://icsme2020.github.io/index.html).*

The code in this repository is used to train and evaluate a deep learning
network capable of recognizing the optimization flags and compiler used in a
compiled binary (currently only `O0`/`O2` and `gcc`/`clang` with our
dataset).
 
This repository contains only the code, pre-trained models can be found at the
[following link](https://zenodo.org/record/3865122#.X0XzttP7T_Q)
  
## Pre-requisites
In order to run the code python 3.7+ is required. Additional dependencies are
listed in the [`requirements.txt`](requirements.txt) file and may be
installed with `pip`.
However, having a GPU supporting CUDA is suggested. This implies installing 
[CUDA drivers](https://developer.nvidia.com/cuda-downloads) and
[cuDNN libraries](https://developer.nvidia.com/cudnn).

## Dataset
The dataset can be found at the 
[following link](https://zenodo.org/record/3865122#.X0XzttP7T_Q). 

In order to
use a custom dataset one can prepare binaries using one of the following
 methods:
 - Extract the various functions from the executables and put them in a `.txt` 
file, one function per line. Each function is encoded by concatenating only the
 instruction opcodes without any spaces.
 - Dump the entire `.text` section in a single `.bin` file. 
 
 In our paper we reported the second method being more efficient than the
  first one.
 
 ## Usage
 The usage of this software consist in the following three parts:
 - Dataset preprocessing
 - Training
 - Evaluation
 
 In the following subsections we explain the basic usage, additional flags
 can be retrieved by running the program with the `-h` or `--help` option.
 
 ### Preprocessing
 Dataset must be preprocessed before training, in order to obtain balanced
 classes and training/validation/testing sets. 
 
For preprocessing the following command should be used:
```bash
python3 optimization-detector.py preprocess  -i <input_folder> -c <class ID> -m <model_dir> 
```
where 
- `<input_folder>` is the folder containing the dataset (`.txt` or `.bin`).
- `<class ID>` is an unique ID chosen by the user to represent the current
 category.
- `<model_dir>` is the directory that will contain the trained model and the
 preprocessed data.
- in case the first encoding for the model was used, the one requiring `.txt
` files, an extra flag `-F true` is required.

Note that this command should be run multiple times, every time with a
 different class and the same model dir, for example like this:
 ```bash
python3 optimization-detector.py preprocess -i gcc-o0/ -c 0 -m model_dir/
python3 optimization-detector.py preprocess -i gcc-o1/ -c 1 -m model_dir/ 
python3 optimization-detector.py preprocess -i clang-o0/ -c 2 -m model_dir/
python3 optimization-detector.py preprocess -i clang-o2/ -c 3 -m model_dir/  
 ```

Finally the following command can be used to check the amount of samples that 
will be used for training, validation and testing

```bash
python3 optimization-detector.py summary -m <model_dir>
```

### Training
Training can be run with the following command after preprocessing:
```bash
python3 optimization-detector.py train -m <model_dir> -n <network_type>
```

where `<network_type>` is one of `lstm` or `cnn`.

An extra folder, `logs/` will be generated inside `<model_dir>` containing
 tensorboard data.

### Evaluation
The evaluation in the paper has been run with the following command:
```bash
python3 optimization-detector.py evaluate -m <model_dir> -o output.csv
```

This will test the classification multiple times, each time increasing the
 input vector length. To test a specific length, add the `-f <value>` flag.
 
 ## Authors

 Davide Pizzolotto <<davidepi@ist.osaka-u.ac.jp>>
 
 Katsuro Inoue <<inoue@ist.osaka-u.ac.jp>>
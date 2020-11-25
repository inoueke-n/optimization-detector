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
compiled binary (`O0`/`O1`/`O2`/`O3`/`Os` and `gcc`/`clang` for both
 `x86_64` and `ARM32` with our
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

This software expects a list of binary files as dataset and can use two
 types of analysis: 
- One expecting a sequence of raw bytes extracted from the `.text` section
of the binary *(default)*.
- One expecting the sequence of opcodes composing a function. This analysis
 requires disassembling before extracting the various opcodes, a quite long
  operation, and is referred in the command line options as *function based*.
 
 ## Usage
 The usage of this software consist in the following four parts:
 - Dataset extraction
 - Dataset preprocessing
 - Training
 - Evaluation
 
 In the following subsections we explain the basic usage. Additional flags
 can be retrieved by running the program with the `-h` or `--help` option.
 
 ### Extraction
 This step is used to extract data from the binary. 
 
 The following command should be used:
 ```
python3 optimization-detector.py extract <input_files> <output_dir>
```
where
- `<input_files>` is the list of binaries.
- `<output_dir>` is the folder where the data should be extracted. For each
 binary a specific file with the same name will be created, with extension
  `.bin` or `.txt` depending on the chosen type of analysis.
 
 By default, the raw data analysis is used. To employ the function based
  analysis, one should add `-F true` as additional flag.
  
 ### Preprocessing
 Dataset must be preprocessed before training, in order to obtain balanced
 classes and training/validation/testing sets. 
 
For preprocessing the following command should be used:
```bash
python3 optimization-detector.py preprocess -c <class ID> <input_folder> <model_dir> 
```
where 
- `<input_folder>` is the folder containing the dataset (`.txt` or `.bin`).
- `<class ID>` is an unique ID chosen by the user to represent the current
 category.
- `<model_dir>` is the directory that will contain the trained model and the
 preprocessed data.
- in case the function based encoding was used when extracting data, 
an extra flag `-F true` is required. This flag effectively filters the files
 based on their extension.

Note that this command should be run multiple times, every time with a
 different class and the same model dir, for example like this:
 ```bash
python3 optimization-detector.py preprocess -c 0 gcc-o0/ model_dir/
python3 optimization-detector.py preprocess -c 1 gcc-o1/ model_dir/ 
python3 optimization-detector.py preprocess -c 2 gcc-o2/ model_dir/
python3 optimization-detector.py preprocess -c 3 gcc-o3/ model_dir/  
 ```

Finally the following command can be used to check the amount of samples that 
will be used for training, validation and testing

```bash
python3 optimization-detector.py summary <model_dir>
```

### Training
Training can be run with the following command after preprocessing:
```bash
python3 optimization-detector.py train -n <network_type> <model_dir>
```

where `<network_type>` is one of `lstm` or `cnn` and `<model_dir>` is the
 folder containing the result of the [preprocess](#preprocessing) operation.

An extra folder, containing tensorboard data, `logs/` will be generated
 inside `<model_dir>`.

### Evaluation
The evaluation in the paper has been run with the following command:
```bash
python3 optimization-detector.py evaluate <model_dir> -o output.csv
```

This will test the classification multiple times, each time increasing the
 input vector length. To test a specific length, add the `-f <value>` flag.
 
 ## Authors

 Davide Pizzolotto <<davidepi@ist.osaka-u.ac.jp>>
 
 Katsuro Inoue <<inoue@ist.osaka-u.ac.jp>>
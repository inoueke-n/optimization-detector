# Optimization Detector
This is the companion code for the paper 
> **Identifying Compiler and Optimization Level in Binary Code from Multiple 
Architectures**
>
> D. Pizzolotto, K. Inoue
>

The code in this repository is used to train and evaluate a deep learning
network capable of recognizing the optimization level and compiler used in a
compiled binary.

With our dataset we tested:
- `O0`/`O1`/`O2`/`O3`/`Os` and `gcc`/`clang` for both `x86_64` and `AArch64`.
- `O0`/`O1`/`O2`/`O3`/`Os` and `gcc` for `RISC-V`, `ARM32`, `PowerPC`,
  `SPARC64` and `MIPS`
 
This repository contains only the code, pre-trained models can be found at the
[following link](https://zenodo.org/record/3865122#.X0XzttP7T_Q)
  
## Pre-requisites
In order to run the code python 3.6+ is required. Additional dependencies are
listed in the [`requirements.txt`](requirements.txt) file and may be
installed with `pip`.
However, having a GPU supporting CUDA is suggested. This implies installing 
[CUDA drivers](https://developer.nvidia.com/cuda-downloads) and
[cuDNN libraries](https://developer.nvidia.com/cudnn).

## Dataset
The manually generated dataset can be found at the 
[following link](https://zenodo.org/record/3865122#.X0XzttP7T_Q).
Alternatively, one can follow the instructions on the 
[dataset generation section](#generation) to generate a gcc-only dataset 
automatically, for any architecture having `gcc`, `g++` and `binutils`
available in the Ubuntu Packages repository.

This software expects a list of binary files as dataset and can use two
 types of analysis: 
- One expecting a sequence of raw bytes extracted from the `.text` section
of the binary *(default)*.
- One expecting the sequence of opcodes composing a function. This analysis
  requires disassembling before extracting the various opcodes, a quite long
  operation, and is referred in the command line options as *encoded*.
  Given the poor results with this second method, we implemented it only for
  the `x86_64` architecture. All the disassembled functions for this method can 
  be found in the archive `amd64-encoded.tar.xz` provided in the dataset.

An [additional file](paper_eval.sh) can be used to replicate our evaluation.
This file should not be run blindly, and is provided only to have an idea of our
overall training approach. Using it in a different system may require some changes.

## Usage

The usage of this software consist in the following four parts:

- Dataset generation
- Dataset extraction
- Dataset preprocessing
- Training
- Evaluation
- Inference

In the following subsections we explain the basic usage. Additional flags can
be retrieved by running the program with the `-h` or `--help` option.

### Generation

We prepared an automated script capable of generating the dataset using any
`gcc` cross compiler available on the 
[Ubuntu Packages repository](https://packages.ubuntu.com/). In this study
we used this script to prepare the `riscv64`, `sparc64`, `powerpc`, `mips` and 
`armhf` architectures. If you retrieved our dataset from zenodo, just 
extract everything and jump to the next section.

Given that compilation results may vary greatly based on the host environment, 
using `docker` to generate the dataset is mandatory.

First create the image using:

```bash
$ docker build -t <image_name> .
```

Then execute the command on the newly created container:

```bash
$ docker run -it <image_name> python3 generate_dataset.py -t "riscv64-linux-gnu" /build 
```

In this command the `-t` parameter specifies which architectures will be built,
and expects a `machine-operatingsystem` tag. This is the same tag that can be 
found in the toolchains available on the *Ubuntu Package Archive*. To build
more than one architecture, one can use `:` to separate them, for example
`"riscv64-linux-gnu:arm-linux-gnueabihf"`. This will build the flags `-O0`, 
`-O1`, `-O2`, `-O3` and `-Os` for each architecture.

**Note**: building requires at least 150GB of free disk available 
(even though the final result will be less than 1GB), and at least 10GB of
system RAM. Expect the building to last a couple of hours for each 
architecture-flag combination.

As soon as the build is finished, one can use the following command to copy out
the results.

```bash
$ docker cp /build/riscv64-gcc-o0.tar.xz <target_directory>
```

where `riscv64` and `o0` should be replaced accordingly with the input 
architecture and optimization level.

At this point, the dataset should be extracted with 

```bash
$ tar xf <archive> -C <target>
```

in order to be used by the next step (ironically called Dataset Extraction 
as well, even though is a different kind of extraction).

### Extraction

This step is used to extract only executable data from the binary.

The following command should be used:

 ```bash
$ python3 optimization-detector.py extract <input_files> <output_dir>
```

where

- `<input_files>` is the list of binaries.
- `<output_dir>` is the folder where the data should be extracted. For each
 binary a specific file with the same name will be created, with extension
  `.bin` or `.txt` depending on the chosen type of analysis.
 
 By default, the raw data analysis is used. To employ the opcode based
  analysis, one should add `--encoded` as additional flag.
  
 ### Preprocessing
 Dataset must be preprocessed before training, in order to obtain balanced
 classes and training/validation/testing sets. 
 
For preprocessing the following command should be used:

```bash
$ python3 optimization-detector.py preprocess -c <class ID> <input_folder> [<input_folder> ...] <model_dir> 
```

where 
- `<input_folder>` is the folder containing the dataset (`.txt` or `.bin`).
- `<class ID>` is an unique ID chosen by the user to represent the current
 category.
- `<model_dir>` is the directory that will contain the trained model and the
 preprocessed data.
- in case the opcode based encoding was used when extracting data, 
an extra flag `--encoded` is required. This flag effectively filters the files
 based on their extension.

Note that this command should be run multiple times, every time with a
 different class and the same model dir, for example like this:

 ```bash
$ python3 optimization-detector.py preprocess --incomplete -c 0 gcc-o0/ clang-o0/ model_dir/
$ python3 optimization-detector.py preprocess --incomplete -c 1 gcc-o1/ clang-o1/ model_dir/ 
$ python3 optimization-detector.py preprocess --incomplete -c 2 gcc-o2/ clang-o2/ model_dir/
$ python3 optimization-detector.py preprocess -c 3 gcc-o3/ clang-o3/ model_dir/  
 ```

The `--incomplete` flag is used to save time by avoiding shuffling and 
duplicate elimination in intermediate steps, but is not strictly necessary.

Finally, the following command can be used to check the amount of samples that 
will be used for training, validation and testing

```bash
$ python3 optimization-detector.py summary <model_dir>
```

### Training
Training can be run with the following command after preprocessing:

```bash
$ python3 optimization-detector.py train -n <network_type> <model_dir>
```

where `<network_type>` is one of `lstm` or `cnn` and `<model_dir>` is the
 folder containing the result of the [preprocess](#preprocessing) operation.

An extra folder, containing tensorboard data, `logs/` will be generated
inside `<model_dir>`.

### Evaluation

The evaluation in the paper has been run with the following command:

```bash
$ python3 optimization-detector.py evaluate -m <model> -o output.csv <dataset_dir>
```

where:

- `<model>` points to the trained `.h5` file
- `<dataset_dir>` points to the directory containing the `test.bin`
  preprocessed dataset

This will test the classification multiple times, each time increasing the
input vector length. To test a specific length, and obtain the confusion
matrix, add the `--confusion <value>` flag.

### Inference

The single-file inference has been run using the following command:
```bash
$ python3 optimization-detector.py infer -m <model> -o output.csv <path-to-file>
```

This command will divide the file in chunks of 2048 bytes each and run the 
inference for each one. Then, the result of each chunk inference will be written
in the file `output.csv`. 
If the `-o output.csv` part is omitted, the average will be reported in stdout.

## Pre-Trained Models
Pre-trained models for every architecture in our dataset can be downloaded from
the [following link](https://sel.ist.osaka-u.ac.jp/people/davidepi/models/).

Note that LSTM models always provide better accuracy (4.5% better on average), 
while CNN models provide faster inference (2x-4x faster).

## Authors

Davide Pizzolotto <<davidepi@ist.osaka-u.ac.jp>>

Katsuro Inoue <<inoue@ist.osaka-u.ac.jp>>

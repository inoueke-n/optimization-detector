import csv
import math
import os
import random
import sys
from typing import List

from termcolor import colored
from tqdm import tqdm

from .binaryds import BinaryDs

MINIMUM_FEATURES: int = 32
csv.field_size_limit(sys.maxsize)


def run_preprocess(input_dir: str, category: int, model_dir: str,
                   function: bool, features: int, split: float,
                   balanced: bool) -> None:
    """
    Performs the preprocessing by adding a category and writes (or updates) the
    binary file containing the dataset on disk
    :param input_dir The folder where the examples for a single category can be
     found
    :param category: The id of the category that will be written
    :param model_dir: Path to the folder where the train.bin and test.bin
    can be found (or will be created)
    :param function: True if this method has a function level granularity
    :param features: How many features (i.e. The number of bytes for each
    example)
    :param split: The ratio training examples over total examples
    :param balanced: True if the produced dataset should have the same
    amount of training/testing/validate samples for each category
    """
    assert (os.path.exists(model_dir))
    train = BinaryDs(os.path.join(model_dir, "train.bin"))
    test = BinaryDs(os.path.join(model_dir, "test.bin"))
    validate = BinaryDs(os.path.join(model_dir, "validate.bin"))
    files = gather_files(input_dir, function)
    print("Loading old dataset... ", end="", flush=True)
    read_dataset(train, function, features)
    read_dataset(test, function, features)
    read_dataset(validate, function, features)
    print(colored("OK", "green", attrs=['bold']), flush=True)
    print("Reading new files... ", flush=True)
    data = read_and_clean_content(files, function, features)
    # Re-mix train and test for accurate duplicates elimination
    # most times this will return just [].
    old_data = train.get(category)
    old_data.extend(test.get(category))
    old_data.extend(validate.get(category))
    data.extend(old_data)
    del old_data
    # now shuffle, remove duplicates and split into train, test validation
    print("Shuffling... ", end="", flush=True)
    random.shuffle(data)
    data = list(set(data))
    split_index = math.floor(len(data) * split)
    new_train_data = data[:split_index]
    data = data[split_index:]
    split_index = math.floor(len(data) * split)
    new_test_data = data[:split_index]
    new_validation_data = data[split_index:]
    train.set(category, new_train_data)
    test.set(category, new_test_data)
    validate.set(category, new_validation_data)
    print(colored("OK", "green", attrs=['bold']), flush=True)
    print("Balancing... ", end="", flush=True)
    if balanced:
        # discard examples as I want also test/val balanced
        train.rebalance(None)
        validate.rebalance(None)
        test.rebalance(None)
        print(colored("OK", "green", attrs=['bold']), flush=True)
    else:
        print(colored("SKIP", "white", attrs=['bold']), flush=True)
    print("Writing... ", end="", flush=True)
    train.write()
    test.write()
    validate.write()
    print(colored("OK", "green", attrs=['bold']), flush=True)


def run_summary(model_dir: str) -> None:
    """
    Gets a summary of the dataset contained in a directory
    :param model_dir: Path to the folder where the train.bin, test.bin and
    validate.bin can be found
    """
    assert (os.path.exists(model_dir))
    train_bin = os.path.join(model_dir, "train.bin")
    test_bin = os.path.join(model_dir, "test.bin")
    validate_bin = os.path.join(model_dir, "validate.bin")
    train = BinaryDs(train_bin)
    assert os.path.exists(train_bin), "Train dataset does not exists!"
    test = BinaryDs(test_bin)
    assert os.path.exists(test_bin), "Test dataset does not exists!"
    val = BinaryDs(validate_bin)
    assert os.path.exists(validate_bin), "Validation dataset does not exists!"
    train.read()
    test.read()
    val.read()
    print(f"Number of classes: {train.get_categories()}")
    if train.get_function_granularity():
        assert test.get_function_granularity()
        assert val.get_function_granularity()
        print("Type: function-based")
    else:
        assert not test.get_function_granularity()
        assert not val.get_function_granularity()
        print("Type: raw values")
    print("--------------------")
    for i in range(0, train.get_categories()):
        print(f"Training examples for class {i}: {len(train.get(i))}")
    assert test.get_categories() == train.get_categories()
    for i in range(0, train.get_categories()):
        print(f"Testing examples for class {i}: {len(test.get(i))}")
    assert val.get_categories() == train.get_categories()
    for i in range(0, train.get_categories()):
        print(f"Validation examples for class {i}: {len(val.get(i))}")


def read_dataset(dataset: BinaryDs, function: bool, features: int) -> None:
    """
    Reads the dataset and assert that is compatible with the requested
    granularity and features.
    :param dataset: The dataset that will be read
    :param function: If function granularity has been requested
    :param features: Number of features requested
    """
    try:
        dataset.read()
        # validate dataset
        if dataset.get_function_granularity() != function:
            raise ValueError("Incompatible granularity between existing "
                             "dataset and requested changes")
        if dataset.get_features() != features:
            raise ValueError("Incompatible number of features between "
                             "existing dataset and requested changes")
    except FileNotFoundError:
        # setup dataset
        dataset.set_features(features)
        dataset.set_function_granularity(function)


def read_and_clean_content(files: List[str], function: bool,
                           features: int) -> List[bytes]:
    """
    Reads the raw files provided by the other program and merge all
    functions of the various files.
    Functions/data with more bytes than the number of features will be split
    into several chunks of features length.
    If function grained is chosen, chunks with less than MINIMUM_FEATURES bytes
    (default 32) will be discarded, otherwise chunks with an amount of bytes
    different than the number of features will be discarded.
    :param files: List of paths to every file that will be processed
    :param function: True if the requested analysis should be function grained
    :param features: The number of features expected
    :return A list of bytes where each element is an example in the category
    """
    if function:
        x = read_files_function(files)
    else:
        x = read_files_raw(files)
    # split in chunks of "features" length
    chunked = []
    for el in x:
        chunks = [el[j:j + features] for j in range(0, len(el), features)]
        chunked.extend(chunks)
    if function:
        # drop elements less than minimum size
        x = list(filter(lambda l: len(l) >= MINIMUM_FEATURES, chunked))
    else:
        # drop elements less than minimum size
        x = list(filter(lambda l: len(l) == features, chunked))
    return x


def gather_files(path: str, function: bool) -> List[str]:
    """
    Finds all files contained in a directory and filter them based on their
    extensions.
    :param path: Path to the folder containing the files or to a single file
    :param function: True if function grained is requested (will parse .txt
    files, .bin otherwise)
    :return A list of paths to every file contained in the folder with .txt
    or .bin extension (based on the function parameter)
    """

    if os.path.isdir(path):
        files = list()
        for _, _, found in os.walk(path):
            for cur_file in found:
                cur_abs = os.path.join(path, cur_file)
                files.append(cur_abs)
    else:
        files = [path]
    if function:
        ext = ".csv"
    else:
        ext = ".bin"
    files = list(filter(lambda x: os.path.splitext(x)[1] == ext, files))
    if len(files) == 0:
        raise FileNotFoundError(f"No files with the correct extension, "
                                "{ext} were found in the given folder")
    return files


def read_files_function(files_list: List[str]) -> List[bytes]:
    """
    Reads all the function opcodes contained in a list of file. Each file is
    expected to be a .csv one, with the opcodes in the field "opcodes" as an
    hex string.

    :param files_list: The list of files that will be parsed
    :return A list of every sample contained in the file, where each sample
    is a sequence of bytes
    """
    functions = list()
    for cur_file in tqdm(files_list, ncols=60):
        with open(cur_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=",", quotechar='"',
                                    quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                data = row["opcodes"]
                encoded_data = bytes.fromhex(data)
                functions.append(encoded_data)
    return functions


def read_files_raw(files_list: List[str]) -> List[bytes]:
    """
    Reads all the raw bytes contained in a file.
    :param files_list: The list of files that will be parsed
    :return A list of every sample contained in the file, where each sample
    is a sequence of bytes
    """
    functions = list()
    for cur_file in tqdm(files_list, ncols=60):
        with open(cur_file, 'rb') as f:
            functions.append(f.read())
    return functions

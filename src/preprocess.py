import math
import os
import random
from typing import List

from .binaryds import BinaryDs


def run_preprocess(input_dir: str, category: int, model_dir: str,
                   function: bool, features: int, split: float,
                   balanced: bool) -> None:
    """
    Perform the preprocessing by adding a category and writes (or updates) the
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
    read_dataset(train, function, features)
    read_dataset(test, function, features)
    read_dataset(validate, function, features)
    data = read_and_clean_content(files, function, features)
    # Re-mix train and test for accurate duplicates elimination
    # most times this will return just [].
    old_data = train.get(category)
    old_data.extend(test.get(category))
    old_data.extend(validate.get(category))
    data.extend(old_data)
    del old_data
    # now shuffle, remove duplicates and split into train, test validation
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
    if balanced:
        # discard examples as I want also test/val balanced
        train.rebalance(None)
        validate.rebalance(None)
        test.rebalance(None)
    train.write()
    test.write()
    validate.write()


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
    functions of the various files
    :param files: List of paths to every file that will be processed
    :param function: True if the requested analysis should be function grained
    :param features: The number of features expected
    :return A list of bytes where each element is an example in the category
    """
    if function:
        x = read_files_function(files)
        x = encode_opcodes(x)
    else:
        x = read_files_raw(files)
        # split in chunks of "features" length
        chunked = []
        for el in x:
            chunked.extend([el[j:j + features]
                            for j in range(0, len(el), features)])
        # drop elements different from features size
        x = list(filter(lambda l: len(l) == features, chunked))
        del chunked
    return x


def encode_opcodes(func_list: List[str]) -> List[bytes]:
    """
    Transform a comma separated list of opcodes (as string bytes) into a
    list of bytes
    :param func_list: The list of opcodes in the form ["DEADC0DE01"]. No
    spaces, commas or single digits allowed (i.e. write 01 instead of 1).
    :return A list of bytes where each element is an example in the category
    """
    func_list = list(map(bytes.fromhex, func_list))
    return func_list


def gather_files(path: str, function: bool) -> List[str]:
    """
    Find all files contained in a directory and filter them based on their
    extensions
    :param path: Path to the folder containing the files
    :param function: True if function grained is requested (will parse .txt
    files, .bin otherwise)
    :return A list of paths to every file contained in the folder with .txt
    or .bin extension (based on the function parameter)
    """
    files = list()
    for _, _, found in os.walk(path):
        for cur_file in found:
            cur_abs = os.path.join(path, cur_file)
            files.append(cur_abs)
    if function:
        ext = ".txt"
    else:
        ext = ".bin"
    files = list(filter(lambda x: os.path.splitext(x)[1] == ext, files))
    if len(files) == 0:
        raise FileNotFoundError(f"No files with the correct extension, "
                                "{ext} were found in the given folder")
    return files


def read_files_function(files_list: List[str]) -> List[str]:
    """
    Read all the function opcodes contained in a file. Every line of the file
    is expected to contain the opcodes.
    :param files_list: The list of files that will be parsed
    :return A list of every sample contained in the file, where each sample
    is a string in the form "DE,AD,C0,DE"
    """
    functions = list()
    for cur_file in files_list:
        with open(cur_file, 'r') as f:
            for cnt, line in enumerate(f):
                line = line.strip('\n')
                if line == "FF," or line == "" or line == "[]":
                    continue
                functions.append(line)
    return functions


def read_files_raw(files_list: List[str]) -> List[bytes]:
    """
    Read all the raw bytes contained in a file.
    :param files_list: The list of files that will be parsed
    :return A list of every sample contained in the file, where each sample
    is a sequence of bytes
    """
    functions = list()
    for cur_file in files_list:
        with open(cur_file, 'rb') as f:
            functions.append(f.read())
    return functions

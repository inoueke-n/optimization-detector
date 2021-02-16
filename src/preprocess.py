import csv
import os
import sys
from typing import List

from termcolor import colored
from tqdm import tqdm

from .binaryds import BinaryDs

MINIMUM_FEATURES: int = 32
csv.field_size_limit(sys.maxsize)


def run_preprocess(input_dir: List[str], category: int, model_dir: str,
                   openc: bool, features: int, balanced: bool,
                   seed: int, incomplete: bool) -> None:
    """
    Performs the preprocessing by adding a category and writes (or updates) the
    binary file containing the dataset on disk
    :param input_dir The folder where the examples for a single category can be
     found
    :param category: The id of the category that will be written
    :param model_dir: Path to the folder where the train.bin and test.bin
    can be found (or will be created)
    :param openc: True if this method has a function opcode encoding
    :param features: How many features (i.e. The number of bytes for each
    example)
    :param balanced: True if the produced dataset should have the same
    amount of training/testing/validate samples for each category
    :param seed: The seed that will be used for shuffling
    :param incomplete: True if the dataset won't be splitted, deduplicated
     or shuffled
    """
    assert (os.path.exists(model_dir))
    train, validate, test = __load_all_into_train(model_dir, features, openc)
    print("Reading and adding new files... ", flush=True)
    files = gather_files(input_dir, openc)
    read_and_add(train, files, category)
    if incomplete:
        print("Deduplicating... ", end="", flush=True)
        print(colored("SKIP", "white", attrs=['bold']), flush=True)
        print("Shuffling... ", end="", flush=True)
        print(colored("SKIP", "white", attrs=['bold']), flush=True)
        print("Balancing... ", end="", flush=True)
        print(colored("SKIP", "white", attrs=['bold']), flush=True)
        print("Splitting... ", end="", flush=True)
        print(colored("SKIP", "white", attrs=['bold']), flush=True)
    else:
        print("Deduplicating... ", end="", flush=True)
        train.deduplicate()
        print(colored("OK", "green", attrs=['bold']), flush=True)
        print("Shuffling... ", end="", flush=True)
        train.shuffle(seed)
        print(colored("OK", "green", attrs=['bold']), flush=True)
        print("Balancing... ", end="", flush=True)
        if balanced:
            train.balance()
            print(colored("OK", "green", attrs=['bold']), flush=True)
        else:
            print(colored("SKIP", "white", attrs=['bold']), flush=True)
        print("Splitting... ", end="", flush=True)
        train.split(validate, 0.5)
        validate.split(test, 0.5)
        print(colored("OK", "green", attrs=['bold']), flush=True)
    print("Finalizing... ", end="", flush=True)
    train.close()
    validate.close()
    test.close()
    print(colored("OK", "green", attrs=['bold']), flush=True)


def __load_all_into_train(model_dir: str, features: int,
                          openc: bool) -> (BinaryDs, BinaryDs, BinaryDs):
    # Load all the files into the train dataset
    print("Loading old dataset... ", end="", flush=True)
    path_train = os.path.join(model_dir, "train.bin")
    path_val = os.path.join(model_dir, "validate.bin")
    path_test = os.path.join(model_dir, "test.bin")
    train = BinaryDs(path_train, features=features, encoded=openc).open()
    test = BinaryDs(path_test, features=features, encoded=openc).open()
    validate = BinaryDs(path_val, features=features, encoded=openc).open()
    train.merge(test)
    train.merge(validate)
    print(colored("OK", "green", attrs=['bold']), flush=True)
    return train, validate, test


def read_and_add(dataset: BinaryDs, files: List[str], category: int) -> None:
    """
    Reads the raw files add them directly to the dataset as examples.
    Functions/data with more bytes than the number of features will be split
    into several chunks of features length.
    If opcode encoding was chosen, chunks with less than MINIMUM_FEATURES bytes
    (default 32) will be discarded, otherwise chunks with an amount of bytes
    different than the number of features will be discarded.
    :param files: List of paths to every file that will be processed.
    :param dataset: dataset where the examples will be added.
    :param category: The category for the current examples.
    """
    buffer = []
    for cur_file in tqdm(files, ncols=60):
        data = list()
        features = dataset.get_features()
        openc = dataset.is_encoded()
        if openc:
            with open(cur_file, 'r') as f:
                reader = csv.DictReader(f, delimiter=",", quotechar='"',
                                        quoting=csv.QUOTE_NONNUMERIC)
                for row in reader:
                    raw_data = row["opcodes"]
                    encoded_data = bytes.fromhex(raw_data)
                    data.append(encoded_data)
        else:
            with open(cur_file, 'rb') as f:
                data.append(f.read())
        # split in chunks of "features" length
        chunked = []
        for el in data:
            chunks = [el[j:j + features] for j in range(0, len(el), features)]
            chunked.extend(chunks)
        if openc:
            # prepad remaining elements and drop ones that are too short
            padded = []
            for element in chunked:
                cur_len = len(element)
                if cur_len >= MINIMUM_FEATURES:
                    missing = features - cur_len
                    padded.append(bytes(missing) + element)
            chunked = padded
        else:
            # drop elements different from feature size
            chunked = list(filter(lambda l: len(l) == features, chunked))
        # append category and add to dataset
        chunked = [(category, x) for x in chunked]
        buffer.extend(chunked)
        if len(buffer) > int(4194304 / (features + 1)):
            # write only when a certain size is reached
            dataset.write(buffer)
            buffer = []
    if len(buffer) > 0:
        # write remaining
        dataset.write(buffer)


def gather_files(paths: List[str], openc: bool) -> List[str]:
    """
    Finds all files contained in a directory and filter them based on their
    extensions.
    :param paths: Paths to the folder containing the files or to a single file
    :param openc: True if opcode based encoding is requested (will parse .csv
    files, .bin otherwise)
    :return A list of paths to every file contained in the folder with .csv
    or .bin extension (based on the function parameter)
    """
    files = []
    for path in paths:
        if os.path.isdir(path):
            cur_files = []
            for _, _, found in os.walk(path):
                for cur_file in found:
                    cur_abs = os.path.join(path, cur_file)
                    cur_files.append(cur_abs)
        else:
            cur_files = [path]
        if openc:
            ext = ".csv"
        else:
            ext = ".bin"
        cur_files = list(
            filter(lambda x: os.path.splitext(x)[1] == ext, cur_files))
        if len(cur_files) == 0:
            raise FileNotFoundError(f"No files with the correct extension, "
                                    "{ext} were found in the given folder")
        else:
            files.extend(cur_files)
    return files

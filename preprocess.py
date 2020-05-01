import itertools
import math
import os
import random

DATASET_NAME = "/mnt/md0/flag_detection/dataset"
PREFIX = "BCCFLD_"
X0_FOLDER = "lfs-clang-O0"
X1_FOLDER = "lfs-clang-O2"
FUNCTION_GRAINED_PREFIX = "-func"
EXECUTABLE_GRAINED_PREFIX = "-whole"
CHUNK_SIZE = 2048
MIN_CHUNK_SIZE = 1


def run_preprocess(input_dir: str, classes: str, model_dir: str):
    # assert validity of classes
    pass


def write_dataset(model_dir, function_grained=True):
    x0, x1 = read_and_clean(DIR, function_grained)
    train, test = gen_train_test(x0, x1, function_grained, cut=0.7,
                                 chunk_size=FEATURES)
    write_binary(train, model_dir + "train.bin", function_grained,
                 chunk_size=FEATURES)
    write_binary(test, model_dir + "test.bin", function_grained,
                 chunk_size=FEATURES)


def gen_train_test(x0_data, x1_data, function_grained, cut=0.7,
                   chunk_size=CHUNK_SIZE):
    """Generate train, tests, X and Y and write them to file"""
    allfun = list()
    for x in x0_data:
        if function_grained:
            allfun.append([x, 0])
        else:
            x = [x[i:i + chunk_size] for i in range(0, len(x), chunk_size)]
            if not x:
                continue
            elif len(x[-1]) > MIN_CHUNK_SIZE:
                x[-1] = x[-1].ljust(chunk_size, b"\0")
            else:
                x = x[:-1]
            for xi in x:
                allfun.append([xi, 0])
    for x in x1_data:
        if function_grained:
            allfun.append([x, 1])
        else:
            x = [x[i:i + chunk_size] for i in range(0, len(x), chunk_size)]
            if not x:
                continue
            elif len(x[-1]) > MIN_CHUNK_SIZE:
                x[-1] = x[-1].ljust(chunk_size, b"\0")
            else:
                x = x[:-1]
            for xi in x:
                allfun.append([xi, 1])
    random.shuffle(allfun)
    train = list()
    test = list()
    index_train = int(math.ceil(len(allfun) * cut))
    for i in range(0, index_train):
        train.append({"x": allfun[i][0], "y": allfun[i][1]})
    for i in range(index_train, len(allfun)):
        test.append({"x": allfun[i][0], "y": allfun[i][1]})
    return train, test


def write_binary(data, filename, function_grained, chunk_size=CHUNK_SIZE):
    """I like binary files """
    if function_grained:
        with open(filename, mode='wb') as f:
            for function in data:
                for opcode in function["x"]:
                    if opcode <= 255:
                        f.write(bytes([opcode]))
                    else:
                        f.write(
                            bytes([(opcode & 0xFF00) >> 8, opcode & 0x00FF]))
                # ends with 0x0F04 if class is 0, 0x0F05 if class is 1
                f.write(bytes([0x0F, 0x04 + function["y"]]))
    else:
        assert (chunk_size == len(data[0]["x"]))
        with open(filename, mode='wb') as f:
            f.write(chunk_size.to_bytes(4, byteorder='big'))
            f.write(len(data).to_bytes(4, byteorder='big'))
            for function in data:
                f.write(function["x"])
                f.write(function["y"].to_bytes(1, byteorder='big'))


def read_binary(filename, function_grained):
    if function_grained:
        return __read_binary_function_grained(filename)
    else:
        return __read_binary_executable_grained(filename)


def __read_binary_executable_grained(filename):
    data = list()
    with open(filename, "rb") as f:
        chunk_size = int.from_bytes(f.read(4), byteorder='big')
        chunk_no = int.from_bytes(f.read(4), byteorder='big')
        for i in range(0, chunk_no):
            data.append({
                "x": list(f.read(chunk_size)),
                "y": int.from_bytes(f.read(1), byteorder='big')
            })
    return data


def __read_binary_function_grained(filename):
    data = list()
    with open(filename, "rb") as f:
        buffer = f.read()
    i = 0
    cur_function = {"x": list(), "y": 0}
    while i < len(buffer):
        byte = buffer[i]
        i += 1
        if byte == 0xFF and buffer[i] == 0xFF:  # special case, invalid opcode
            cur_function["x"].append(0xFFFF)
            i += 1
        elif byte != 0x0F:
            cur_function["x"].append(byte)
        else:
            byte2 = buffer[i]
            i += 1
            if byte2 == 0x04:
                cur_function["y"] = 0
                data.append(cur_function)
                cur_function = {"x": list(), "y": 0}
            elif byte2 == 0x05:
                cur_function["y"] = 1
                data.append(cur_function)
                cur_function = {"x": list(), "y": 0}
            else:
                cur_function["x"].append((byte << 8) | byte2)
    return data


def read_and_clean(dir, function_grained):
    """Read the raw files provided by the other program and merge all
    functions of the various files"""
    x0_files, x1_files = gather_files(dir, function_grained)
    x0_funcs = read_files_content(x0_files, function_grained)
    x1_funcs = read_files_content(x1_files, function_grained)
    if function_grained:
        x0_funcs = split_function(x0_funcs)
        x1_funcs = split_function(x1_funcs)
        x0_funcs = cleanup(x0_funcs)
        x1_funcs = cleanup(x1_funcs)
    return x0_funcs, x1_funcs


def split_function(func_list):
    """ Split functions into a list of opcodes """
    return [list(filter(None, item.split(","))) for item in func_list]


def cleanup(func_list):
    """ Transform from string to numbers"""
    x = [[int(opcode, 16) for opcode in fn] for fn in func_list]
    x.sort()
    x = list(x for x, _ in itertools.groupby(x))  # remove duplicates
    return x


def gather_files(dir: str, function_grained):
    x0_files = list()
    x1_files = list()
    x0_abs_dir = os.path.join(dir, "dataset", X0_FOLDER)
    x1_abs_dir = os.path.join(dir, "dataset", X1_FOLDER)
    if function_grained:
        x0_abs_dir += FUNCTION_GRAINED_PREFIX
        x1_abs_dir += FUNCTION_GRAINED_PREFIX
    else:
        x0_abs_dir += EXECUTABLE_GRAINED_PREFIX
        x1_abs_dir += EXECUTABLE_GRAINED_PREFIX
    for _, _, files in os.walk(x0_abs_dir):
        for cur_file in files:
            cur_abs = os.path.join(x0_abs_dir, cur_file)
            x0_files.append(cur_abs)
    for _, _, files in os.walk(x1_abs_dir):
        for cur_file in files:
            cur_abs = os.path.join(x1_abs_dir, cur_file)
            x1_files.append(cur_abs)
    return x0_files, x1_files


def read_files_content(files_list, function_grained):
    functions = list()
    for cur_file in files_list:
        if function_grained:
            with open(cur_file, 'r') as f:
                for cnt, line in enumerate(f):
                    line = line.strip('\n')
                    if line == "FF," or line == "" or line == "[]":
                        continue
                    functions.append(line)
        else:
            with open(cur_file, 'rb') as f:
                functions.append(f.read())
    return functions

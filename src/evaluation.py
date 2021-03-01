import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model

from src.binaryds import BinaryDs
from src.datagenerator import DataGenerator


def run_evaluation(data_dir: str, model_path: str, file: str,
                   seed: int, fixed: int, bs: int) -> None:
    """
    Runs the evaluation on the test dataset and reports accuracy or confusion
    matrix.
    The evaluation will be normally run by evaluating inputs with only 1
    feature, then 2 features, then 3 and so on up to the maximum number of
    features. The increment in features is not linear.
    :param data_dir: string pointing to the folder containing the test.bin
    file (generated with the run_preprocess function)
    :param model_path: string pointing to the .h5 keras model of the network.
    If empty will default to data_dir/model.h5
    :param file: string pointing to the file that will contain the evaluation (
    will be a .csv so add the extension by yourself)
    :param seed: seed that will be used for training
    :param fixed: if different from 0 only this specific number of features
    will be tested and the confusion matrix reported. If equals to 0,
    every feature from 1 to max.
    :param bs: batch size
    """
    if seed == 0:
        seed = int(time.time())
    np.random.seed(seed)  # This is actually useless in this method...
    assert os.path.exists(data_dir), "Model directory does not exists!"
    if model_path == "":
        model_path = os.path.join(data_dir, "model.h5")
    assert os.path.exists(model_path), f"Model {model_path} does not exists!"
    output_dir = os.path.abspath(os.path.join(file, os.pardir))
    assert os.access(output_dir, os.W_OK), "Output folder is not writable"
    test_bin = os.path.join(data_dir, "test.bin")
    assert os.path.exists(test_bin), "Test dataset does not exists!"
    if fixed == 0:
        evaluate_incremental(bs, file, model_path, test_bin)
    else:
        evaluate_confusion(bs, file, fixed, model_path, test_bin)


def evaluate_incremental(bs: int, file: str, model_path: str,
                         test_bin) -> None:
    """
    Evaluates the accuracy incrementally (first only 1 feature, then 3, then 5)
    :param bs: batch size
    :param file: file where to write the accuracy (.csv)
    :param model_path: string pointing to the .h5 keras model of the network.
    If empty will default to data_dir/model.h5
    :param test_bin: path to the test dataset that will be used
    """
    cut = 1
    test = BinaryDs(test_bin, read_only=True).open()
    model = load_model(model_path)
    features = test.get_features()
    with open(file, "w") as f:
        f.write("features,accuracy\n")
    while cut <= features:
        print(f"Evaluating {cut}")
        generator = DataGenerator(test, bs, fake_pad=True, pad_len=cut)
        score = model.evaluate(generator)
        with open(file, "a") as f:
            f.write(f"{cut},{score[1]}\n")
        if cut < 24:
            cut = cut + 2
        elif cut < 80:
            cut = cut + 22
        elif cut < 256:
            cut = cut + 33
        elif cut < 500:
            cut = cut + 61
        elif cut < features:
            cut = cut + 129
            cut = min(cut, features)
        else:
            break
    test.close()


def get_expected(bs: int, test):
    """
    Get the expected predictions for an entire dataset
    :param bs: batch size
    :param test:
    :return:
    """
    total = test.get_examples_no()
    iterations = int(total / bs)
    remainder = total % bs
    result = []
    for batch in range(iterations):
        data = test.read(batch * bs, bs)
        y, _ = zip(*data)
        result.extend(y)
    if remainder != 0:
        data = test.read(iterations * bs, remainder)
        y, _ = zip(*data)
        result.extend(y)
    return np.array(result, dtype=np.int8)


def evaluate_confusion(bs: int, file: str, fixed: int, model_path: str,
                       test_bin) -> None:
    """
    Evaluates the confusion matrix for a given number of features
    :param bs: batch size
    :param file: file where the confusion matrix will be written
    :param fixed: number of features to be considered
    :param model_path: string pointing to the .h5 keras model of the network.
    If empty will default to data_dir/model.h5
    :param test_bin: path to the test dataset that will be used
    """
    test = BinaryDs(test_bin, read_only=True).open()
    binary = test.get_categories() <= 2
    model = load_model(model_path)
    generator = DataGenerator(test, bs, fake_pad=True, pad_len=fixed,
                              predict=True)
    expected = get_expected(bs, test)
    predicted = model.predict(generator, verbose=1)
    if binary:
        predicted = np.round(predicted).flatten().astype(np.int8)
    else:
        predicted = np.argmax(predicted, axis=1)
    matrix = np.array(tf.math.confusion_matrix(expected, predicted))
    with open(file, "w") as f:
        np.savetxt(f, X=matrix, fmt="%d")
    test.close()

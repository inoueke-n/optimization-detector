from __future__ import print_function

import os
import time
from typing import Union

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout, LeakyReLU
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence
from tensorflow.python import confusion_matrix, SparseAdd, SparseTensor

from src.binaryds import BinaryDs

# name of the model on the disk
MODEL_NAME = "model.h5"


def run_train(model_dir: str, seed: int, network: str) -> None:
    """
    Trains the model
    :param model_dir: string pointing to the folder containing the train.bin
    and validation.bin files (generated with the run_preprocess function)
    :param seed: seed that will be used for training
    :param network: either "dense", "lstm" or "cnn", to choose which
    function to train
    """
    if seed == 0:
        seed = int(time.time())
    assert os.path.exists(model_dir), "Model directory does not exists!"
    train_bin = os.path.join(model_dir, "train.bin")
    validate_bin = os.path.join(model_dir, "validate.bin")
    assert os.path.exists(train_bin), "Train dataset does not exists!"
    assert os.path.exists(validate_bin), "Validation dataset does not exists!"
    train = BinaryDs(train_bin)
    validate = BinaryDs(validate_bin)
    train.read()
    validate.read()
    model_path = os.path.join(model_dir, MODEL_NAME)
    if os.path.exists(model_path):
        print("Loading previously created model")
        model = load_model(model_path)
    else:
        print(f"Creating new {network} model")
        if network == "dense":
            model = model_dense(train.get_categories(), train.get_features())
        elif network == "lstm":
            model = model_lstm(train.get_categories(), train.get_features())
        elif network == "cnn":
            model = model_cnn(train.get_categories(), train.get_features())
        else:
            raise ValueError("The parameter `network` was not specified (or "
                             "not chosen between dense/lstm/cnn")
    print(model.summary())
    np.random.seed(seed)
    if train.get_function_granularity():
        # functions are already variable sized so no need to pad
        fake_pad = False
    else:
        # pad the regularly sized chunks instead
        fake_pad = True
    x_train, y_train = generate_sequences(train, fake_pad)
    x_val, y_val = generate_sequences(validate, fake_pad)
    x_train = sequence.pad_sequences(x_train, maxlen=train.features,
                                     padding="pre", truncating="pre",
                                     value=0, dtype="int32")
    x_val = sequence.pad_sequences(x_val, maxlen=validate.features,
                                   padding="pre", truncating="pre",
                                   value=0, dtype="int32")

    checkpoint = ModelCheckpoint(filepath=model_path,
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True)

    tensorboard_logs = os.path.join(model_dir, 'logs')
    os.makedirs(tensorboard_logs, exist_ok=True)
    tensorboad = TensorBoard(log_dir=tensorboard_logs,
                             histogram_freq=1,
                             write_graph=True,
                             write_images=True,
                             update_freq=5,
                             embeddings_freq=1)
    early_stopper = EarlyStopping(monitor="val_loss",
                                  min_delta=0.001,
                                  patience=3,
                                  mode="auto")

    model.fit(x_train, y_train, epochs=40, batch_size=256,
              validation_data=(x_val, y_val),
              callbacks=[tensorboad, checkpoint, early_stopper])


def generate_sequences(data: BinaryDs, fake_pad: bool) -> (np.array, np.array):
    """
    Generates the pairs (X, y) that will be used during the training.
    More specifically generates y, shuffle X and randomly remove data from X
    otherwise the network won't learn how to deal with padding.
    :param data: binary dataset containing the samples
    :param fake_pad: true if padding should be added
    :return: (X, y) as np.arrays. X shape will be (samples, features),
    y will be (samples) or (samples, categories) depending if binary or
    multiclass classification
    """
    x = []
    y = []  # using lists since I don't know the final size beforehand
    assert data.get_features() > 31, "Minimum number of features is 32"
    cat_no = data.get_categories()
    for i in range(0, cat_no):
        samples = data.get(i)
        if samples:
            # multiclass, generate array with prediction for each class
            if cat_no > 2:
                expected = [0.0] * data.get_categories()
                expected[i] = 1.0
                expected = [expected for _ in range(0, len(samples))]
                y.extend(expected)
            # binary, single value 0 or 1 is sufficient
            else:
                expected = [i] * len(samples)
                y.extend(expected)
            # keras does not like bytearrays, so int list then
            # also, randomly cut a portion of them, so network learns to deal
            # with padding
            if fake_pad:
                cut = np.random.randint(31, data.get_features(), len(samples))
                samples_int = [list(sample)[:cut[idx]]
                               for idx, sample in enumerate(samples)]
            else:
                samples_int = [list(sample) for sample in samples]
            x.extend(samples_int)
    x = np.array(x)
    y = np.array(y)
    assert len(x) == len(y), "Something went wrong... different X and y len"
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    return x, y


def model_dense(classes: int, features: int) -> Sequential:
    """
    Generates the dense network model.
    :param classes: Number of categories to be recognized
    :param features: Number of features in input
    :return: The keras model
    """
    model = Sequential()
    model.add(Input(shape=features, ))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    if classes <= 2:
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy",
                      optimizer=Adam(1e-3),
                      metrics=["binary_accuracy"])
    else:
        model.add(Dense(classes, activation="softmax"))
        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(1e-3),
                      metrics=["categorical_accuracy"])
    return model


def model_lstm(classes: int, features: int) -> Sequential:
    """
        Generates the LSTM network model.
        :param classes: Number of categories to be recognized
        :param features: Number of features in input
        :return: The keras model
        """
    embedding_size = 256
    embedding_length = 128
    model = Sequential()
    model.add(Embedding(embedding_size, embedding_length,
                        input_length=features))
    model.add(LSTM(256))
    if classes <= 2:
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy",
                      optimizer=Adam(1e-3),
                      metrics=["binary_accuracy"])
    else:
        model.add(Dense(classes, activation="softmax"))
        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(1e-3),
                      metrics=["categorical_accuracy"])
    return model


def model_cnn(classes: int, features: int) -> Sequential:
    """
    Generates the CNN network model.
    :param classes: Number of categories to be recognized
    :param features: Number of features in input
    :return: The keras model
    """
    embedding_size = 256
    embedding_length = 128
    model = Sequential()
    model.add(Embedding(embedding_size, embedding_length,
                        input_length=features))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same',
                     strides=1, activation=None))
    model.add(Conv1D(filters=32, kernel_size=5, padding='same',
                     strides=2, activation=None))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling1D(pool_size=2, padding="same"))

    model.add(Conv1D(filters=64, kernel_size=3, padding='same',
                     strides=1, activation=None))
    model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                     strides=2, activation=None))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling1D(pool_size=2, padding="same"))

    model.add(Conv1D(filters=128, kernel_size=3, padding='same',
                     strides=1, activation=None))
    model.add(Conv1D(filters=128, kernel_size=5, padding='same',
                     strides=2, activation=None))
    model.add(LeakyReLU(alpha=0.01))
    model.add(MaxPooling1D(pool_size=2, padding="same"))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    if classes <= 2:
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy",
                      optimizer=Adam(1e-3),
                      metrics=["binary_accuracy"])
    else:
        model.add(Dense(classes, activation="softmax"))
        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(1e-3),
                      metrics=["categorical_accuracy"])
    return model


def run_evaluation(model_dir: str, file: str, stop: int, incr: int,
                   seed: int, fixed: int) -> None:
    """
    Run the evaluation on the test dataset and reports the confusion matrix.
    The evaluation will be normally run by evaluating inputs with only 1
    feature, then 2 features, then 3 and so on up to the number specified by
    the parameter stop.
    :param model_dir: string pointing to the folder containing the train.bin
    and validation.bin files (generated with the run_preprocess function)
    :param file: string pointing to the file that will contain the evaluation (
    will be a .csv so add the extension by yourself)
    :param stop: in case of increasingly padded values, when to stop
    :param incr: in case of increasingly padded values, the increment for
    each step. if 0, the increase will be not linear (I specially tailored
    this increment for my needs so there isn't a specific function)
    :param seed: seed that will be used for training
    :param fixed: if different from 0 only this specific number of features
    will be tested. If equals to 0, every feature from 1 to stop.
    """
    if seed == 0:
        seed = int(time.time())
    np.random.seed(seed)  # This is actually useless in this method...
    if fixed != 0:
        cut = fixed
    else:
        cut = 1
    assert os.path.exists(model_dir), "Model directory does not exists!"
    model_path = os.path.join(model_dir, MODEL_NAME)
    output_path = os.path.join(model_dir, file)
    test_bin = os.path.join(model_dir, "test.bin")
    assert os.path.exists(test_bin), "Test dataset does not exists!"
    test = BinaryDs(test_bin)
    test.read()
    categories = test.get_categories()
    function = test.get_function_granularity()
    features = test.get_features()
    x, y = generate_sequences(test, fake_pad=False)
    limit = stop
    if limit == 0:
        limit = test.get_features()
    while cut <= limit:
        print(f"Evaluating {cut}")
        nx, ny = cut_dataset(x, y, function, cut)
        matrix = evaluate_nn(model_path, nx, ny, categories, features)
        # binary, write confusion matrix
        matrix = np.asarray(matrix)
        if categories <= 2:
            with open(output_path, "a") as f:
                f.write(str(cut) + ",")
                f.write(str(matrix[0][0]) + ",")
                f.write(str(matrix[0][1]) + ",")
                f.write(str(matrix[1][0]) + ",")
                f.write(str(matrix[1][1]) + "\n")
        # multiclass, calculate just accuracy
        else:
            correct = 0
            for i in range(0, matrix.shape[0]):
                correct += matrix[i][i]
            total = sum(sum(matrix))
            accuracy = 0
            if total != 0:
                accuracy = correct / total
            with open(output_path, "a") as f:
                f.write(str(cut) + ",")
                f.write(str(accuracy) + "\n")
        if fixed != 0:
            return
        elif incr == 0:
            if cut < 24:  # more accurate evaluation where required
                cut = cut + 2
            elif cut < 80:
                cut = cut + 4
            elif cut < 256:
                cut = cut + 22
            elif cut < 500:
                cut = cut + 61
            elif cut < features:
                cut = cut + 129
                cut = min(cut, features)
            else:
                cut = 0xFFFFFFFFFFFFFFFF
        else:
            cut = cut + 1


def cut_dataset(x: np.array, y: np.array, function: bool,
                cut: int) -> (np.array, np.array):
    """
    Replace part of the input with zeroes.
    :param x: inputs
    :param y: expected predictions
    :param function: true if the opcode-based evaluation should be used
    :param cut: how many features to keep
    """
    if function:
        nx = []
        ny = []
        for i in range(0, len(x)):
            if len(x[i]) <= cut:
                nx.append(x[i])
                ny.append(y[i])
        return np.asarray(nx), np.asarray(ny)
    else:
        nx = np.empty((x.shape[0], cut))
        for i in range(0, len(x)):
            nx[i] = x[i][:cut]
        return np.asarray(nx), y


def evaluate_nn(model_path: str, x_test: np.array, y_test: np.array,
                classes: int, features: int) -> Union[SparseTensor, SparseAdd]:
    """
    Actual inference.
    :param model_path: string pointin to the model.h5 file
    :param x_test: input vectors
    :param y_test:  expected prediction
    :param classes: number of categories to predict
    :param features: number of features in input
    :return: The confusion matrix. Its shape depends if multiclass or single
    class.
    """
    model = load_model(model_path)
    x_test = sequence.pad_sequences(x_test, maxlen=features, dtype="int32",
                                    padding="pre", truncating="pre", value=0)
    yhat_classes = model.predict_classes(x_test, verbose=1, batch_size=256)
    if classes > 2:
        y_test = np.argmax(y_test, axis=1)
        matrix = confusion_matrix(y_test, yhat_classes, num_classes=classes)
    else:
        yhat_classes = yhat_classes[:, 0]
        matrix = confusion_matrix(y_test, yhat_classes, num_classes=2)
    print(matrix)
    return matrix

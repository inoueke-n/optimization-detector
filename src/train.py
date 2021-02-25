import os
import time

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from .binaryds import BinaryDs
# name of the model on the disk
from .datagenerator import DataGenerator

MODEL_NAME = "model.h5"


def run_train(data_dir: str, seed: int, network: str, bs: int) -> None:
    """
    Trains the model
    :param data_dir: string pointing to the folder containing the train.bin
    and validation.bin files (generated with the run_preprocess function)
    :param seed: seed that will be used for training
    :param network: either "dense", "lstm" or "cnn", to choose which
    function to train
    :param bs: batch size. If None it will be automatically determined
    """
    if seed == 0:
        seed = int(time.time())
    assert os.path.exists(data_dir), "Model directory does not exists!"
    train_bin = os.path.join(data_dir, "train.bin")
    validate_bin = os.path.join(data_dir, "validate.bin")
    assert os.path.exists(train_bin), "Train dataset does not exists!"
    assert os.path.exists(validate_bin), "Validation dataset does not exists!"
    train = BinaryDs(train_bin, read_only=True).open()
    validate = BinaryDs(validate_bin, read_only=True).open()
    model_dir = os.path.join(data_dir, network)
    model_path = os.path.join(model_dir, MODEL_NAME)
    if os.path.exists(model_path):
        print("Loading previously created model")
        model = load_model(model_path)
    else:
        print(f"Creating new {network} model")
        os.makedirs(model_dir, exist_ok=True)
        if network == "lstm":
            model = model_lstm(train.get_categories(), train.get_features())
        elif network == "cnn":
            model = model_cnn(train.get_categories(), train.get_features())
        else:
            train.close()
            validate.close()
            raise ValueError("The parameter `network` was not specified (or "
                             "not chosen between lstm/cnn")
    print(model.summary())
    np.random.seed(seed)
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
    gen_train = DataGenerator(train, bs)
    gen_val = DataGenerator(validate, bs)
    model.fit(gen_train,
              validation_data=gen_val,
              epochs=40,
              callbacks=[tensorboad, checkpoint, early_stopper])
    train.close()
    validate.close()


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
#
# This file was used for hyperparameters evaluation and mostly with
# hardcoded values. Don't read it or try to execute it please as I didn't
# have time to clean it in time for the deadline (and it's useless for the
# actual evaluation).
# Btw it also contains the old CNN model without the leaky relu
#

import os
import time

import numpy as np
from kerastuner import HyperModel, Hyperband
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence

from src.binaryds import BinaryDs
from src.learning import generate_sequences


def run_tuner(model_dir: str, seed: int) -> None:
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
    np.random.seed(seed)
    x_train, y_train = generate_sequences(train)
    x_val, y_val = generate_sequences(validate)
    x_train = sequence.pad_sequences(x_train, maxlen=train.features,
                                     padding="post", truncating="post")
    x_val = sequence.pad_sequences(x_val, maxlen=validate.features,
                                   padding="post", truncating="post")
    hypermodel = TuneLSTM(train.get_categories(), train.get_features())
    tuner = Hyperband(hypermodel, max_epochs=10, objective='val_accuracy',
                      seed=32000, executions_per_trial=2)
    tuner.search_space_summary()
    tuner.search(x_train, y_train, epochs=10, batch_size=256,
                 validation_data=(x_val, y_val))
    tuner.results_summary()


class TuneCNN(HyperModel):
    def __init__(self, classes, features):
        super().__init__()
        self.features = features
        self.classes = classes

    def build(self, hp):
        embedding_size = 256

        model = Sequential()
        model.add(Embedding(embedding_size,
                            hp.Int('embedding_len', min_value=8, max_value=256,
                                   step=32, default=64),
                            input_length=self.features))
        model.add(Conv1D(
            filters=hp.Choice('num_filters1', values=[32, 64], default=64),
            kernel_size=hp.Choice('kernel_size1', values=[3, 5, 7], default=3),
            padding='same',
            activation='relu'))
        model.add(Conv1D(
            filters=hp.Choice('num_filters2', values=[32, 64], default=64),
            kernel_size=hp.Choice('kernel_size2', values=[3, 5, 7], default=3),
            padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2, padding="valid"))
        model.add(Conv1D(
            filters=hp.Choice('num_filters3', values=[32, 64], default=64),
            kernel_size=hp.Choice('kernel_size3', values=[3, 5, 7], default=3),
            padding='same', activation='relu'))
        model.add(Conv1D(
            filters=hp.Choice('num_filters4', values=[32, 64], default=64),
            kernel_size=hp.Choice('kernel_size4', values=[3, 5, 7], default=3),
            padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2, padding="valid"))
        model.add(Conv1D(
            filters=hp.Choice('num_filters5', values=[32, 64], default=64),
            kernel_size=hp.Choice('kernel_size5', values=[3, 5, 7], default=3),
            padding='same', activation='relu'))
        model.add(Conv1D(
            filters=hp.Choice('num_filters6', values=[32, 64], default=64),
            kernel_size=hp.Choice('kernel_size6', values=[3, 5, 7], default=3),
            padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2, padding="valid"))
        model.add(Flatten())
        model.add(Dense(hp.Int('units', min_value=8, max_value=128, step=32,
                               default=32), activation="relu"))
        model.add(Dense(self.classes, activation="softmax"))
        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(hp.Float(
                          'learning_rate', min_value=1e-4, max_value=1e-2,
                          sampling="LOG", default=1e-3
                      )),
                      metrics=["accuracy"])
        return model


class TuneLSTM(HyperModel):
    def __init__(self, classes, features):
        super().__init__()
        self.features = features
        self.classes = classes

    def build(self, hp):
        embedding_size = 256
        model = Sequential()
        model.add(Embedding(embedding_size,
                            hp.Int('embedding_len', min_value=8, max_value=256,
                                   step=32, default=64),
                            input_length=self.features))
        model.add(LSTM(hp.Int('lstm', min_value=16, max_value=512,
                              step=64)))
        model.add(Dense(self.classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(hp.Float(
                          'learning_rate', min_value=1e-4, max_value=1e-2,
                          sampling="LOG", default=1e-3
                      )),
                      metrics=['accuracy'])
        return model

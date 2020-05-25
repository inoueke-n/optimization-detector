from __future__ import print_function

import os
import time

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence
from tensorflow.python import confusion_matrix
from tensorflow.python.keras.callbacks import EarlyStopping

from src.binaryds import BinaryDs

MODEL_NAME = "model.h5"


def run_train(model_dir: str, seed: int, use_lstm: bool = False) -> None:
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
        model = load_model(model_path)
        print("Loading previous model")
    elif train.get_categories() <= 2:
        if use_lstm:
            model = binary_lstm(train.get_features())
        else:
            model = binary_cnn(train.get_features())
    else:
        if use_lstm:
            model = multiclass_lstm(train.get_categories(),
                                    train.get_features())
        else:
            model = multiclass_cnn(train.get_categories(),
                                   train.get_features())
    print(model.summary())
    np.random.seed(seed)
    x_train, y_train = generate_sequences(train, fake_pad=True)
    x_val, y_val = generate_sequences(validate, fake_pad=True)
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


def binary_lstm(features: int) -> Sequential:
    embedding_size = 256
    embedding_length = 128
    model = Sequential()
    model.add(Embedding(embedding_size, embedding_length,
                        input_length=features))
    model.add(LSTM(256))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(1e-3),
                  metrics=['binary_accuracy'])
    return model


def binary_cnn(features: int) -> Sequential:
    embedding_size = 256
    embedding_length = 128
    model = Sequential()
    model.add(Embedding(embedding_size, embedding_length,
                        input_length=features))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same',
                     activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=7, padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding="valid"))
    model.add(Conv1D(filters=64, kernel_size=7, padding='same',
                     activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding="valid"))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same',
                     activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=5, padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding="valid"))
    model.add(Flatten())
    model.add(Dense(72, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(1e-3),
                  metrics=["binary_accuracy"])
    return model


def multiclass_lstm(classes: int, features: int) -> Sequential:
    embedding_size = 256
    embedding_length = 128
    model = Sequential()
    model.add(Embedding(embedding_size, embedding_length,
                        input_length=features))
    model.add(LSTM(256))
    model.add(Dense(classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-3),
                  metrics=['categorical_accuracy'])
    return model


def multiclass_dense(classes: int, features: int) -> Sequential:
    model = Sequential()
    model.add(Input(shape=features, ))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-3),
                  metrics=['categorical_accuracy'])
    return model


def multiclass_cnn(classes: int, features: int) -> Sequential:
    embedding_size = 256
    embedding_length = 128
    model = Sequential()
    model.add(Embedding(embedding_size, embedding_length,
                        input_length=features))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same',
                     activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=7, padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding="valid"))
    model.add(Conv1D(filters=64, kernel_size=7, padding='same',
                     activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding="valid"))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same',
                     activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=5, padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2, padding="valid"))
    model.add(Flatten())
    model.add(Dense(72, activation="relu"))
    model.add(Dense(classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(1e-3),
                  metrics=["categorical_accuracy"])
    return model


def run_evaluation(model_dir: str, file: str, stop: int, incr: int,
                   seed: int, fixed: int) -> None:
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
    while cut < limit:
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
                classes: int, features: int):
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

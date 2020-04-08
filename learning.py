from __future__ import print_function

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, \
    TensorBoard
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, \
    Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.python import confusion_matrix

from preprocess import *

BATCH_SIZE = 32


def generate_sequences(data):
    indices = list(range(0, len(data)))
    random.shuffle(indices)
    x = list()
    y = list()
    for i in indices:
        x.append(data[i]["x"])
        y.append(data[i]["y"])
    return np.asarray(x), np.asarray(y)


def train_network(layers, X_train, y_train, X_test, y_test, model_path,
                  pad_length):
    X_train = sequence.pad_sequences(X_train, maxlen=pad_length)
    X_validation = sequence.pad_sequences(X_test, maxlen=pad_length)
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(model.summary())
    else:
        model = layers

    # callbacks
    checkpoint = ModelCheckpoint(filepath=model_path,
                                 verbose=1,
                                 save_best_only=True)
    # early_stopper = EarlyStopping(monitor='val_accuracy',
    #                               min_delta=0.01,
    #                               patience=1,
    #                               verbose=0,
    #                               mode='auto',
    #                               baseline=None,
    #                               restore_best_weights=False)
    tensorboard_logs = os.path.abspath(os.path.join(model_path, os.pardir))
    tensorboard_logs = os.path.join(tensorboard_logs, 'logs')
    os.makedirs(tensorboard_logs, exist_ok=True)
    tensorboad = TensorBoard(log_dir=tensorboard_logs,
                             histogram_freq=1,
                             write_graph=True,
                             write_images=True,
                             update_freq=10000)

    model.fit(X_train,
              y_train,
              epochs=10,
              batch_size=BATCH_SIZE,
              validation_data=(X_validation, y_test),
              callbacks=[tensorboad, checkpoint])


def binary_plain_LSTM(X_train, y_train, X_test=None, y_test=None,
                      model_path="model.h5", embedding_size=65536,
                      pad_length=1024):
    np.random.seed(32000)
    embedding_length = 64

    model = Sequential()
    model.add(Embedding(embedding_size, embedding_length,
                        input_length=pad_length))
    model.add(LSTM(256))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    train_network(model, X_train, y_train, X_test, y_test, model_path,
                  pad_length)


def binary_convolutional_LSTM(X_train, y_train, X_test=None, y_test=None,
                              model_path="model.h5", embedding_size=65536,
                              pad_length=1024):
    np.random.seed(32000)
    embedding_length = 64
    model = Sequential()
    model.add(Embedding(embedding_size, embedding_length,
                        input_length=pad_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(256))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    train_network(model, X_train, y_train, X_test, y_test, model_path,
                  pad_length)


def evaluate_nn(model_path, X_test, y_test, pad_length):
    model = load_model(model_path)
    yhat_classes = model.predict_classes(
        sequence.pad_sequences(X_test, maxlen=pad_length), verbose=1)
    yhat_classes = yhat_classes[:, 0]
    matrix = confusion_matrix(y_test, yhat_classes)
    print(matrix)
    return matrix

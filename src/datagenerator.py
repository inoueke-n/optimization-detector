from typing import List, Tuple

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence

from src.binaryds import BinaryDs

LN100 = 2 * np.log(10)


class DataGenerator(keras.utils.Sequence):

    def __init__(self, dataset: BinaryDs, batch_size: int,
                 predict: bool = False,
                 fake_pad: bool = False, pad_len: int = 0):
        self.dataset: BinaryDs = dataset
        self.batch_size = batch_size
        self.indices: List[int] = []
        self.fake_pad = fake_pad
        self.pad_len = pad_len
        self.predict = predict
        self.len = 0
        self.remainder = 0
        self.__init_len()
        self.on_epoch_end()

    def __init_len(self):
        """
        Sets self.len and self.remainder (used at init time)
        :return:
        """
        self.len = int(self.dataset.get_examples_no() / self.batch_size)
        self.remainder = self.dataset.get_examples_no() % self.batch_size
        if self.remainder > 0:
            self.len += 1

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        real_index = self.indices[index]
        if self.remainder > 0 and real_index == self.len - 1:
            amount = self.remainder
        else:
            amount = self.batch_size
        data = self.dataset.read(real_index * self.batch_size, amount)
        return self.__generate_sequences(data)

    def on_epoch_end(self):
        self.indices = np.arange(self.len)
        if not self.predict:
            np.random.shuffle(self.indices)

    def __generate_sequences(self, data: List[Tuple[int, bytes]]):
        """
        Generates the pairs (X, y) that will be used during the training.
        More specifically generates y and shuffle X.
        If fake_pad is true, randomly removes data from X. This is useful in case
        the training samples have always the same amount of features, but during
        inference this number may change.
        :param data: binary dataset containing the samples
        :return: (X, y) as np.arrays. X shape will be (samples, features),
        y will be (samples) or (samples, categories) depending if binary or
        multiclass classification
        """
        cats = self.dataset.get_categories()
        y, x = zip(*data)
        if cats > 2:
            y = keras.utils.to_categorical(y, num_classes=cats)
        else:
            y = [[y] for y in y]
        # keras does not like bytearrays, so int list then
        # cut a portion of example so network learns to deal with padding
        if not self.dataset.is_encoded() and self.fake_pad:
            # amount of removed data randomly decided
            if self.pad_len == 0:
                limit = self.dataset.features - 32
                # 99% values should be between 0 and limit
                elambda = LN100 / limit
                beta = 1 / elambda
                d = np.random.default_rng().exponential(beta, size=len(x))
                # clamping destroys the distribution, not a big deal
                cut = np.array(np.floor(np.clip(d, 0, limit)),
                               dtype=np.int32)
                x = [list(sample)[:-cut[idx]] for idx, sample in enumerate(x)]
            # amount of removed data is a fixed value
            elif self.dataset.features != self.pad_len:
                cut = np.full(len(x), self.dataset.features - self.pad_len)
                x = [list(sample)[:-cut[idx]] for idx, sample in enumerate(x)]
            else:
                x = [list(sample) for sample in x]
        # keep only encoded examples of `pad_len` length
        elif self.dataset.is_encoded() and self.pad_len != 0:
            new_x = []
            for sample in x:
                padded = 0
                for byte in sample:
                    if byte != 0x00:
                        break
                    else:
                        padded += 1
                if self.dataset.get_features() - padded <= self.pad_len:
                    # shorter than requested, keep intact
                    new_x.append(sample)
                else:
                    # longer than requested, cut it to the requested len
                    new_x.append(sample[-self.pad_len:])
            x = [list(sample) for sample in new_x]
        # keep everything without removing data
        else:
            x = [list(sample) for sample in x]
        x = np.array(x)
        x = sequence.pad_sequences(x, maxlen=self.dataset.features,
                                   padding="pre", truncating="pre",
                                   value=0, dtype="int32")
        y = np.array(y)
        assert len(x) == len(y), \
            "Something went wrong... different X and y len"
        if self.predict:
            return x
        else:
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x = x[indices]
            y = y[indices]
            return x, y

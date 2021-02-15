from typing import List, Tuple

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence

from src.binaryds import BinaryDs


class DataGenerator(keras.utils.Sequence):

    def __init__(self, dataset: BinaryDs, batch_size: int,
                 fake_pad: bool = True):
        self.dataset: BinaryDs = dataset
        self.batch_size = batch_size
        self.indices: List[int] = []
        self.fake_pad = fake_pad
        self.len = int(self.dataset.get_examples_no() / self.batch_size)
        self.on_epoch_end()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        real_index = self.indices[index]
        if real_index == self.len:
            amount = self.dataset.get_examples_no() % self.batch_size
        else:
            amount = self.batch_size
        data = self.dataset.read(real_index * self.batch_size, amount)
        return self.__generate_sequences(data)

    def on_epoch_end(self):
        self.indices = np.arange(self.len)
        np.random.shuffle(self.indices)

    def __generate_sequences(self, data: List[Tuple[int, bytes]]) -> (
            np.array, np.array):
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
            pass
        # keras does not like bytearrays, so int list then
        # also, randomly cut a portion of them, so network learns to deal
        # with padding
        if not self.dataset.is_encoded() and self.fake_pad:
            limit = self.dataset.features - 31
            # calculate the lambda so the maximum value we want is at 4 sd
            # of course we can get higher than that, but it is highly unlikely
            # and clamping does not break the training too much
            elambda = 4/limit
            exp = np.random.default_rng().exponential(1/elambda, size=len(x))
            # clamping destroys the distribution, but it's not a big deal
            exp = np.array(np.floor(np.clip(exp, 0, limit)), dtype=np.int32)
            with open('/tmp/samples.txt', 'a') as file:
                np.savetxt(fname=file, X=exp, delimiter="\n", fmt='%05d')
            x = [list(sample)[:-exp[idx]] for idx, sample in enumerate(x)]
        else:
            x = [list(sample) for sample in x]
        x = np.array(x)
        x = sequence.pad_sequences(x, maxlen=self.dataset.features,
                                   padding="pre", truncating="pre",
                                   value=0, dtype="int32")
        y = np.array(y)
        assert len(x) == len(y), \
            "Something went wrong... different X and y len"
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]
        return x, y

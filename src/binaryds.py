import hashlib
import os
import random
from typing import BinaryIO, Optional, List, Tuple

HEADER_SIZE = 8
MAGIC = 0x27
BLOCK_SIZE = 4194304


class BinaryDs:
    """
    Class used to contain a dataset in a binary form, where each example
    belongs to a given category.

    File structure of the binary:
    1 bytes -> magic
    1 bit -> 0: data is opcode based, 1:data is raw based
    7 bit -> number of categories (max 127 ofc)
    2 bytes -> number of features for each example
    4 bytes -> number of examples
    All the examples in the form {label(1 byte)+data}
    """

    def __init__(self, path: str, read_only: bool = False,
                 features: int = 2048, encoded: bool = True) -> None:
        """
        Constructor

        :param path: path to the binary dataset
        :param read_only: True if the dataset will be open in read only mode
        :param features: number of features that will be used for each example
        :param encoded: true if the contained data will be a raw dump, so not
        an opcode based encoding.
        (This has no effect on the dataset itself, but it is used as a double
        check to avoid mixing data from dataset encoded in different ways)
        """
        self.magic: int = MAGIC
        self.encoded: bool = encoded
        self.modified: bool = False  # true if some write has been performed
        self.path: str = path
        self.features: int = features
        self.examples: int = 0
        self.ro: bool = read_only
        self.file: Optional[BinaryIO] = None

    def open(self):
        """
        Opens a Binary dataset for editing. Creates it if not existing iff
        read_only was set to False in the constructor.
        :return: The created dataset
        """
        if not os.path.exists(self.path):
            if self.ro:
                raise PermissionError("Could not create file (Read only flag)")
            else:
                self.file = open(self.path, "wb+")
                self.file.write(self.magic.to_bytes(1, byteorder="little"))
                if self.encoded:
                    self.file.write(b'\x80')
                else:
                    self.file.write(b'\x00')
                self.file.write(self.features.to_bytes(2, byteorder="little"))
                self.file.write(self.examples.to_bytes(4, byteorder="little"))
        elif self.ro and os.access(self.path, os.R_OK):
            self.file = open(self.path, "rb")
            self.__read_and_check_existing()
        elif not self.ro and os.access(self.path, os.W_OK):
            self.file = open(self.path, "rb+")
            self.__read_and_check_existing()
        else:
            raise PermissionError()
        return self

    def close(self) -> None:
        """
        Closes an open dataset.
        """
        if self.file:
            if self.modified:
                self.__write_max_cats()
            self.file.close()
            self.file = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    # determine the number of categories inside the dataset
    # if the dataset contains an example for 0, and for 2, the categories will
    # be 3 (even though 1 is missing)
    def __calc_max_cat(self) -> int:
        cats = 0
        self.file.seek(HEADER_SIZE, os.SEEK_SET)
        for i in range(self.examples):
            data = self.file.read(self.features + 1)
            cat = data[0]
            if cat >= cats:
                cats = cat + 1
        return cats

    # read the max cat value inside the binary on disk
    def __read_max_cats(self) -> int:
        self.file.seek(1, os.SEEK_SET)
        data = int.from_bytes(self.file.read(1), byteorder="little")
        return data & 0x7F

    def __read_encoding(self) -> bool:
        self.file.seek(1, os.SEEK_SET)
        data = int.from_bytes(self.file.read(1), byteorder="little")
        return data & 0x80 > 0

    # writes the max cat value inside the binary on disk
    def __write_max_cats(self):
        cats = self.__calc_max_cat()
        self.file.seek(1, os.SEEK_SET)
        data = (int(self.encoded) << 7) | (cats & 0x7F)
        self.file.write(data.to_bytes(1, byteorder="little"))

    def __read_and_check_existing(self) -> None:
        # Reads the data from the existing dataset file
        # Checks for consistency with the expected encoding and feature size
        # If the file is open for reading, just update features and encoding
        self.file.seek(0, os.SEEK_SET)
        if int.from_bytes(self.file.read(1), byteorder="little") != MAGIC:
            self.file.close()
            self.file = None
            raise IOError(f"File {self.path} was not created by this "
                          f"application.")
        encoded = (int.from_bytes(self.file.read(1),
                                  byteorder="little") & 0x80) > 0
        features = int.from_bytes(self.file.read(2), byteorder="little")
        self.examples = int.from_bytes(self.file.read(4), byteorder="little")
        # consistency check
        if not self.ro and self.encoded != encoded:
            self.file.close()
            self.file = None
            raise IOError("The existing file has a different encoding type")
        else:
            self.encoded = self.__read_encoding()
        if not self.ro and self.features != features:
            self.file.close()
            self.file = None
            raise IOError("The existing file has a different number of "
                          "features")
        else:
            self.features = features

    def is_encoded(self) -> bool:
        """
        Returns the type of encoding used on the file.
        0 for Raw based, 1 for Opcode based
        """
        return self.encoded

    def get_categories(self) -> int:
        """
        Returns the number of categories used in the dataset.
        This function considers the highest value for each examples: if the
        dataset contains two examples, one with category 0 and one with
        category 2, this function will return 3 despite the category 1 being
        missing.
        """
        return self.__read_max_cats()

    def get_features(self) -> int:
        """
        Returns the number of features used in the dataset.
        """
        return self.features

    def get_examples_no(self) -> int:
        """
        Returns the number of examples contained in the dataset.
        """
        return self.examples

    def write(self, data: List[Tuple[int, bytes]],
              update_categories: bool = False) -> None:
        """
        Writes a list of examples in the file.
        Throws ValueError if the tuple has a different length compared to the
        one passed in the constructor.
        :param data: A tuple (category id, data) that will be written.
        :param update_categories: True if the max number of categories should
        be updated. This will be done in any case when closing the file.
        """
        for val in data:
            if len(val[1]) != self.features:
                raise ValueError("The input example has a wrong length")
        offset = HEADER_SIZE + (self.features + 1) * self.examples
        self.examples += len(data)
        self.file.seek(4, os.SEEK_SET)
        self.file.write(self.examples.to_bytes(4, byteorder="little"))
        self.file.seek(offset, os.SEEK_SET)
        for val in data:
            self.file.write(val[0].to_bytes(1, byteorder="little"))
            self.file.write(val[1])
        self.modified = True
        if update_categories:
            self.__write_max_cats()

    def read(self, index: int, amount: int = 1) -> List[Tuple[int, bytes]]:
        """
        Reads some examples from the dataset.
        Raises IndexError if index+amount is higher than the number of
        available examples.
        :param index: The starting index of the examples to read (0-based)
        :param amount: The number of examples that will be read
        :return: A list of tuples (category id, data), each tuple being an
        example
        """
        if index + amount > self.examples:
            raise IndexError
        else:
            retval = []
            offset = HEADER_SIZE + (self.features + 1) * index
            self.file.seek(offset, os.SEEK_SET)
            for i in range(0, amount):
                cat = int.from_bytes(self.file.read(1), byteorder="little")
                data = self.file.read(self.features)
                retval.append((cat, data))
            return retval

    def shuffle(self, seed=None) -> None:
        """
        Shuffles in place the dataset.
        :param seed: Seed that will be used for the RNG
        """
        random.seed(seed)
        for j_index in range(self.examples)[::-1]:
            k_index = int(random.random() * j_index)
            if j_index != k_index:
                offset_j = HEADER_SIZE + (self.features + 1) * j_index
                self.file.seek(offset_j, os.SEEK_SET)
                j = self.file.read(self.features + 1)
                offset_k = HEADER_SIZE + (self.features + 1) * k_index
                self.file.seek(offset_k, os.SEEK_SET)
                k = self.file.read(self.features + 1)
                self.file.seek(offset_k, os.SEEK_SET)
                self.file.write(j)
                self.file.seek(offset_j, os.SEEK_SET)
                self.file.write(k)

    def balance(self) -> None:
        """
        Balances the dataset. This method will use less memory and be more
        efficient when the dataset is randomized.
        """
        cats = []
        # use counting sort to record how many examples for each class
        self.file.seek(HEADER_SIZE, os.SEEK_SET)
        for i in range(self.examples):
            data = self.file.read(self.features + 1)
            cat = data[0]
            while len(cats) <= cat:
                cats.append(0)
            cats[cat] += 1
        if len(cats) == 0 or len(cats) == 1:
            return  # no point in balancing when there's nothing
        minval = min(cats)
        cats = [x - minval for x in cats]
        stored = []
        # now: remove examples from the end and store them in memory until
        # balance is reached, throwing away the unbalanced ones. Then write
        # back the one in memory.
        # this is n^2 under the assumption that categories are small
        self.file.seek(0, os.SEEK_END)
        while sum(cats) != 0:
            self.file.seek(-self.features - 1, os.SEEK_CUR)
            data = self.file.read(self.features + 1)
            self.file.seek(-self.features - 1, os.SEEK_CUR)
            if cats[data[0]] == 0:
                # need this example for balance
                stored.append((data[0], data[1:]))
            else:
                cats[data[0]] -= 1  # throw example away
            self.examples -= 1
        self.write(stored)

    def truncate(self, left=0) -> None:
        """
        Remove all examples from file
        """
        feature_size = self.features + 1
        self.file.truncate(HEADER_SIZE + feature_size * left)
        self.file.seek(4, os.SEEK_SET)
        self.examples = left
        self.file.write(self.examples.to_bytes(4, byteorder="little"))

    def merge(self, other) -> None:
        """
        Removes all the content from other and puts it into self
        """
        if self.is_encoded() != other.is_encoded():
            raise IOError("To merge two datasets they must have the same "
                          "encoding")
        if self.get_features() != other.get_features():
            raise IOError("To merge two datasets they must have the same "
                          "features")
        examples_no = other.examples
        if examples_no > 0:
            features_size = self.features + 1
            amount = int(BLOCK_SIZE / features_size)
            iterations = int(examples_no / amount)
            for i in range(iterations):
                read = other.read(i * amount, amount)
                self.write(read)
            remainder = examples_no % amount
            if remainder > 0:
                read = other.read(iterations, remainder)
                self.write(read)
            # remove from other file and update elements amount
            other.truncate()

    def split(self, other, ratio: float) -> None:
        """
        Removes part of self and put it to other.
        The variable ration is #examples_kept/#examples_given
        """
        if self.is_encoded() != other.is_encoded():
            raise IOError("To merge two datasets they must have the same "
                          "encoding")
        if self.get_features() != other.get_features():
            raise IOError("To merge two datasets they must have the same "
                          "features")

        examples_no = int(self.examples * ratio)
        if examples_no > 0:
            features_size = self.features + 1
            amount = int(BLOCK_SIZE / features_size)
            iterations = int(examples_no / amount)
            for _ in range(iterations):
                read = self.read(self.examples - amount, amount)
                other.write(read)
                self.truncate(self.examples - amount)

            remainder = examples_no % amount
            if remainder > 0:
                read = self.read(self.examples - remainder, remainder)
                other.write(read)
                self.truncate(self.examples - remainder)

    def deduplicate(self) -> None:
        """
        Removes all duplicates from the current binary.
        The order of data may change.
        """
        feature_size = self.features + 1
        chunk_size = BLOCK_SIZE + feature_size  # around 4 MiB ;)
        chunk_size = int(chunk_size - (chunk_size % feature_size))
        chunk_elements = int(chunk_size / feature_size)
        chunks_no = int(self.examples / chunk_elements)
        hashes = set()
        toremove = set()
        chunk_index = 0
        for _ in range(chunks_no):
            self.__calculate_hashes_from_chunk(chunk_elements, chunk_index,
                                               hashes, toremove)
            chunk_index += 1
        remainder = self.examples % chunk_elements
        if remainder > 0:
            self.__calculate_hashes_from_chunk(remainder, chunk_index,
                                               hashes, toremove)
        del hashes
        # iterate every element from beginning to end
        for i in range(self.examples-1):
            # deal with updating of the self.examples variable
            if i >= self.examples:
                break
            # if I have to remove current element
            if i in toremove:
                # pick an element from end, candidate for elimination
                j = self.examples - 1
                while j > i and j in toremove:
                    self.examples -= 1
                    j = self.examples - 1
                # there is an actual replacement
                if j != i:
                    self.file.seek(HEADER_SIZE + feature_size * j, os.SEEK_SET)
                    data = self.file.read(feature_size)
                    self.file.seek(HEADER_SIZE + feature_size * i, os.SEEK_SET)
                    self.file.write(data)
                    self.examples = j
                # there is no replacement, simply delete last entry
                else:
                    self.examples = i
        # update examples
        self.file.seek(4, os.SEEK_SET)
        self.file.write(self.examples.to_bytes(4, byteorder="little"))
        self.__write_max_cats()

    def __calculate_hashes_from_chunk(self, chunk_elements, chunk_index,
                                      hashes, toremove):
        # internal method used in the deduplicate function
        data = self.read(chunk_index * chunk_elements, chunk_elements)
        for count, elem in enumerate(data):
            index = chunk_index * chunk_elements + count
            cur_hash = hashlib.sha1(elem[1]).digest()
            if cur_hash in hashes:
                toremove.add(index)
            else:
                hashes.add(cur_hash)

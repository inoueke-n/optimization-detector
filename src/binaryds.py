import itertools
from io import BufferedReader, BufferedWriter
from typing import Dict, List


class BinaryDs:
    """
    Class used to contain a dataset in a binary form, where each example belongs to a given category.

    File structure of the binary:
    1 bytes -> magic (0x26)
    1 bytes -> 0: function grained, 1: raw data grained
    2 bytes -> number of features
    2 bytes -> number of predicted categories (The maximum number appearing in the dictionary)
    2 bytes -> number of chunks (The actual number of categories contained in the dictionary)
    ----- FOR EACH CHUNK ----
    2 bytes -> category id
    8 bytes -> size of the chunk in bytes (excluding these 10 bytes)
    All the examples
    -------------------------
    """

    def __init__(self, path: str) -> None:
        """
        Constructor

        Parameters
        ----------
        path: str
            A path to the file that will be managed by this object
        """
        self.magic: int = 0x26
        self.function: bool = False
        self.path: str = path
        self.data: Dict[int, List[bytes]] = {}
        self.features: int = 2048
        self.categories: int = 0

    def read(self) -> None:
        """
        Read the binary file set in the constructor and load its content into the object.
        If the file is not found nothing happens and the default settings are used.
        """
        self.__init__(self.path)
        try:
            with open(self.path, "rb") as f:
                if int.from_bytes(f.read(1), byteorder="big") != self.magic:
                    raise IOError(f"File {self.path} was not created by this application.")
                self.function = int.from_bytes(f.read(1), byteorder="big")
                self.features = int.from_bytes(f.read(2), byteorder="big")
                self.categories = int.from_bytes(f.read(2), byteorder="big")
                chunks_no = int.from_bytes(f.read(2), byteorder="big")
                for _ in range(0, chunks_no):
                    self.__read_chunk(f)
        except FileNotFoundError:
            pass  # no problem, just create the file later while saving

    # read a chunk from the file representing a specific category
    def __read_chunk(self, file: BufferedReader) -> None:
        chunk_id = int.from_bytes(file.read(2), byteorder="big")
        chunk_size = int.from_bytes(file.read(8), byteorder="big")
        data = file.read(chunk_size)
        if self.function:
            data = data.split(b"\x0F\x04")
            data = list(filter(lambda x: x, data))  # remove eventual empty elements resulting from split
        else:
            data = [data[i:i + self.features] for i in range(0, len(data), self.features)]
        self.data[chunk_id] = data

    def write(self) -> None:
        """
        Writes all the contents of this object to a single binary file
        """
        with open(self.path, "wb") as f:
            chunks_no = len(self.data)
            f.write(self.magic.to_bytes(1, byteorder="big"))
            f.write(self.function.to_bytes(1, byteorder="big"))
            f.write(self.features.to_bytes(2, byteorder="big"))
            f.write(self.categories.to_bytes(2, byteorder="big"))
            f.write(chunks_no.to_bytes(2, byteorder="big"))
            for key in self.data:
                self.__write_chunk(key, f)

    # Write a single category to file
    def __write_chunk(self, chunk_id: int, file: BufferedWriter) -> None:
        samples = self.data[chunk_id]
        if samples:
            if self.function:
                samples = [x + b"\x0F\x04" for x in samples]
            samples = bytearray(list(itertools.chain.from_iterable(samples)))
            file.write(chunk_id.to_bytes(2, byteorder="big"))
            file.write(len(samples).to_bytes(8, byteorder="big"))
            file.write(samples)

    def get(self, cat_id: int) -> List[bytes]:
        """
        Returns a category contained inside this object, identified by a specific id
            
        Parameters
        ----------
        cat_id: int
            The id of the category that will be returned
        """
        if cat_id in self.data:
            return self.data[cat_id]
        else:
            return []

    def set(self, cat_id: int, data: List[bytes]) -> None:
        """
        Add a set of examples belonging to a specific category to this object.
        Existing examples will be discarded.

        Parameters
        ----------
        cat_id: int
            The id of the category that will be set
        data: List[bytes]
            The examples that will be set
        """
        self.__check_consistency(data)
        if cat_id > self.categories:
            self.categories = cat_id
        self.data[cat_id] = data

    # Check examples consistency (in the number of features) before adding them
    def __check_consistency(self, data: List[bytes]) -> None:
        if self.function:
            for element in data:
                assert len(
                    element) <= self.features, f"Expected {self.features} features, but {len(element)} were provided"
        else:
            for element in data:
                assert len(
                    element) == self.features, f"Expected {self.features} features, but {len(element)} were provided"

    def set_features(self, val: int) -> None:
        """
        Set the number of features of the examples. This will discard all the examples previously loaded.
        Note that for `raw` granularity all the examples MUST have this size. For `func` granularity this is the maximum
        length allowed

        Parameters
        ---------
        val: int
            The number of features that will be used.
        """
        self.features = val
        self.data = {}
        self.categories = 0

    def get_features(self) -> int:
        """
        Get the number of features expected in the examples.

        Returns
        ------
        int : The number of features
        """
        return self.features

    def get_categories(self) -> int:
        """
        Get the highest id of any categories found inside the examples in this object

        Returns
        -------
        int: The highest category_id expected
        """
        return self.categories

    def set_function_granularity(self, val: bool) -> None:
        """
        Set the granularity of the current examples. True for `function`, False for `raw`. Note that this will discard
        all previously loaded examples.
        The default value for this is False

        Parameters
        ---------
        val : bool
            True if `function` granularity should be used, False otherwise
        """
        self.function = val
        self.data = {}
        self.categories = 0

    def get_function_granularity(self) -> bool:
        """
        Get the granularity of the current examples

        Returns
        ------
        bool : True if the granularity is set to `function`, False otherwise
        """
        return self.function

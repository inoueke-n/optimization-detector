import os
import random
import shutil
import tempfile
from unittest import TestCase

from src.binaryds import BinaryDs

PREFIX = "BCCFLT_"


class TestBinaryDs(TestCase):
    tmpdir: str = None
    data_raw = [
        (0, b"\x93\xE7\x32\x89\x1B\x08\x01\x65\x31\xA6\x64\x18\x3B\xA2"),
        (1, b"\xDE\xDE\xF8\x1B\x83\x0E\xBF\xD0\x59\x04\x7C\x76\x33\x6F"),
        (2, b"\xB5\xAF\x4D\x2B\xBD\xDA\x32\x65\x61\x62\x90\x35\xEA\x7E"),
    ]
    data_raw2 = [
        (0, b"\xC0\xBB\x3B\x1E\x55\xB7\x45\x17\xC2\x83\x86\x33\xCE\xAB"),
        (2, b"\xA7\x11\x87\xB2\x7D\xDF\x6E\x03\x13\x2F\xEA\x64\xCD\x3C"),
        (1, b"\x2B\x30\x54\x22\xB3\xE2\xEA\x75\xC2\xF5\x2E\x74\xDB\xEC"),
        (0, b"\xC1\xC9\x9B\x88\x33\x48\x20\x4B\xC6\x1B\x38\xDF\x9E\xBD"),
        (2, b"\x83\x52\xBE\x52\xC1\x5B\xB9\xD4\x1E\xFD\x2C\xA4\x0D\x63"),
        (0, b"\xD5\x33\x59\xC9\x62\x71\x48\x00\x62\x7A\x0D\x89\x3C\x73"),
        (1, b"\xAF\xDA\x83\x9E\x46\x46\x79\xBA\x20\x1D\x56\x38\x98\x8D"),
        (0, b"\xAA\xDE\xC4\xE3\x27\x70\x52\x84\x4C\xBB\x3B\x57\x7C\x34")
    ]

    @classmethod
    def setUpClass(self):
        systmpdir = tempfile.gettempdir()
        self.tmpdir = tempfile.mkdtemp(prefix=PREFIX, dir=systmpdir)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.tmpdir)

    def test_open_readonly_not_existing(self):
        file = os.path.join(self.tmpdir, "readonly_not_existing.bin")
        with self.assertRaises(PermissionError):
            BinaryDs(file, True).open()

    # write some raw data with the wrong number of features. Assert error
    def test_write_wrong_number_features(self):
        file = os.path.join(self.tmpdir, "write_wrong_features.bin")
        with BinaryDs(file) as dataset:
            with self.assertRaises(ValueError):
                dataset.write(self.data_raw)

    # open existing file with the wrong encoding
    def test_wrong_encoding(self):
        file = os.path.join(self.tmpdir, "wrongenc.bin")
        dataset = BinaryDs(file, encoded=False).open()
        dataset.close()
        with self.assertRaises(IOError):
            BinaryDs(file, encoded=True).open()

    # open existing file with the wrong encoding (readonly). should succeed
    def test_wrong_encoding_readonly(self):
        file = os.path.join(self.tmpdir, "wrongenc_readonly.bin")
        dataset = BinaryDs(file, encoded=False).open()
        dataset.close()
        with BinaryDs(file, encoded=True, read_only=True) as dataset:
            self.assertFalse(dataset.is_encoded())

    # open existing file with the wrong encoding
    def test_open_wrong_features(self):
        file = os.path.join(self.tmpdir, "open_wrong_features.bin")
        dataset = BinaryDs(file, features=1024).open()
        dataset.close()
        with self.assertRaises(IOError):
            BinaryDs(file, features=2048).open()

    # open existing file with the wrong encoding (readonly). should succeed
    def test_open_wrong_features_readonly(self):
        file = os.path.join(self.tmpdir, "open_wrong_features_readonly.bin")
        dataset = BinaryDs(file, features=1024).open()
        dataset.close()
        with BinaryDs(file, features=2048, read_only=True) as dataset:
            self.assertEqual(dataset.get_features(), 1024)

    # Write a file. Then read it. Assert the content is ok
    def test_read_write(self):
        file = os.path.join(self.tmpdir, "rw.bin")
        binary = BinaryDs(file, features=14).open()
        binary.write(self.data_raw)
        binary.close()
        with BinaryDs(file, features=14, read_only=True) as dataset:
            read = dataset.read(0, len(self.data_raw))
        self.assertEqual(read, self.data_raw)

    # try to write into a read-only dataset
    def test_write_to_ro(self):
        file = os.path.join(self.tmpdir, "write_ro.bin")
        dataset = BinaryDs(file, features=14).open()
        dataset.close()
        with BinaryDs(file, features=14, read_only=True) as dataset:
            with self.assertRaises(IOError):
                dataset.write(self.data_raw)

    # Write a file multiple times. Then read a part of it
    def test_multi_read_write(self):
        file = os.path.join(self.tmpdir, "rwmulti.bin")
        with BinaryDs(file, features=14) as binary:
            binary.write(self.data_raw)
        with BinaryDs(file, features=14) as binary:
            binary.write(self.data_raw2)
        expected = [self.data_raw[2]] + self.data_raw2[:3]
        with BinaryDs(file, features=14, read_only=True) as dataset:
            read = dataset.read(2, 4)
        self.assertEqual(read, expected)

    # Write a file and then shuffle it (in place)
    def test_shuffle(self):
        seed = 32000
        # assert that the order is the expected one
        random.seed(seed)
        expected_order = [4, 0, 1, 6, 3, 7, 5, 2]
        file = os.path.join(self.tmpdir, "shuffle.bin")
        with BinaryDs(file, features=14) as binary:
            binary.write(self.data_raw2)
            binary.shuffle(seed)
        with BinaryDs(file, features=14) as binary:
            results = binary.read(0, binary.examples)
        for res_idx, exp_idx in enumerate(expected_order):
            self.assertEqual(results[res_idx], self.data_raw2[exp_idx])

    # Write a file and then balance it (in place)
    def test_balance(self):
        file = os.path.join(self.tmpdir, "balance.bin")
        with BinaryDs(file, features=14) as binary:
            binary.write(self.data_raw2)
            binary.balance()
        with BinaryDs(file, features=14) as binary:
            results = binary.read(0, binary.examples)
        expected = self.data_raw2[:5] + [self.data_raw2[6]]
        self.assertEqual(results, expected)

    # Asserts the correct number of features. file already open
    def test_get_encoding(self):
        file_raw = os.path.join(self.tmpdir, "encoding_raw.bin")
        file_op = os.path.join(self.tmpdir, "encoding_op.bin")
        with BinaryDs(file_raw, encoded=True) as dataset_raw:
            self.assertTrue(dataset_raw.is_encoded())
        with BinaryDs(file_op, encoded=False) as dataset_op:
            self.assertFalse(dataset_op.is_encoded())

    # Asserts the correct number of features. file already open
    def test_get_features(self):
        file = os.path.join(self.tmpdir, "features.bin")
        with BinaryDs(file, features=14) as dataset:
            self.assertEqual(dataset.get_features(), 14)

    # Asserts the correct number of examples. file already open
    def test_get_examples(self):
        file = os.path.join(self.tmpdir, "examples.bin")
        with BinaryDs(file, features=14) as dataset:
            self.assertEqual(dataset.get_examples_no(), 0)
            dataset.write(self.data_raw)
            self.assertEqual(dataset.get_examples_no(), 3)

    # Asserts exception when merging different encodings
    def test_merge_different_encoding(self):
        file_op = os.path.join(self.tmpdir, "merge_op.bin")
        file_raw = os.path.join(self.tmpdir, "merge_raw.bin")
        with BinaryDs(file_op, encoded=False, features=14) as ds_op:
            ds_op.write(self.data_raw)
            with BinaryDs(file_raw, encoded=True, features=14) as ds_raw:
                ds_raw.write(self.data_raw)
                with self.assertRaises(IOError):
                    ds_raw.merge(ds_op)

    # Asserts exception when merging different features length
    def test_merge_different_features(self):
        file14 = os.path.join(self.tmpdir, "merge_f14.bin")
        file2k = os.path.join(self.tmpdir, "merge_f2048.bin")
        with BinaryDs(file2k) as ds2k:
            with BinaryDs(file14, features=14) as ds14:
                ds14.write(self.data_raw)
                with self.assertRaises(IOError):
                    ds2k.merge(ds14)

    # Asserts all examples are correctly removed
    def test_truncate_all(self):
        file = os.path.join(self.tmpdir, "truncate.bin")
        dataset = BinaryDs(file, features=14).open()
        dataset.close()
        empty_size = os.path.getsize(file)
        with BinaryDs(file, features=14) as dataset:
            dataset.write(self.data_raw2)
        self.assertGreater(os.path.getsize(file), empty_size)
        with BinaryDs(file, features=14) as dataset:
            dataset.truncate()
        self.assertEqual(os.path.getsize(file), empty_size)

    # Asserts all examples but one are correctly removed
    def test_truncate_but_one(self):
        file = os.path.join(self.tmpdir, "truncate_1.bin")
        with BinaryDs(file, features=14) as dataset:
            dataset.write(self.data_raw2)
        self.assertGreater(dataset.get_examples_no(), 0)
        with BinaryDs(file, features=14) as dataset:
            dataset.truncate(left=1)
        self.assertEqual(dataset.get_examples_no(), 1)

    # Asserts correct merge
    def test_merge(self):
        file1 = os.path.join(self.tmpdir, "mergeA.bin")
        file2 = os.path.join(self.tmpdir, "mergeB.bin")
        dataset1 = BinaryDs(file1, features=14).open()
        dataset1.write(self.data_raw)
        dataset2 = BinaryDs(file2, features=14).open()
        dataset2.write(self.data_raw2)
        self.assertEqual(dataset1.get_examples_no(), 3)
        self.assertEqual(dataset2.get_examples_no(), 8)
        dataset1.merge(dataset2)
        self.assertEqual(dataset1.get_examples_no(), 11)
        self.assertEqual(dataset2.get_examples_no(), 0)
        self.assertEqual(dataset1.read(0, 11), self.data_raw + self.data_raw2)
        dataset1.close()
        dataset2.close()

    def test_split(self):
        file1 = os.path.join(self.tmpdir, "splitA.bin")
        file2 = os.path.join(self.tmpdir, "splitB.bin")
        dataset1 = BinaryDs(file1, features=14).open()
        dataset1.write(self.data_raw2)
        dataset2 = BinaryDs(file2, features=14).open()
        self.assertEqual(dataset1.get_examples_no(), 8)
        self.assertEqual(dataset2.get_examples_no(), 0)
        dataset1.split(dataset2, 0.5)
        self.assertEqual(dataset1.get_examples_no(), 4)
        self.assertEqual(dataset2.get_examples_no(), 4)
        self.assertEqual(dataset1.read(0, 4), self.data_raw2[:4])
        self.assertEqual(dataset2.read(0, 4), self.data_raw2[4:])
        dataset1.close()
        dataset2.close()

    def test_deduplicate(self):
        file = os.path.join(self.tmpdir, "deduplicate.bin")
        expected = [self.data_raw[0]] + \
                   [self.data_raw2[-1]] + \
                   self.data_raw[1:] + \
                   self.data_raw2[:-1]
        with BinaryDs(file, features=14) as dataset:
            dataset.write([self.data_raw[0]])
            dataset.write(self.data_raw)
            dataset.write(self.data_raw2)
            dataset.write(self.data_raw)
            dataset.write(self.data_raw2)
        with BinaryDs(file, features=14) as dataset:
            dataset.deduplicate()
            data = dataset.read(0, 11)
            self.assertEqual(data, expected)

    def test_update_categories(self):
        file = os.path.join(self.tmpdir, "categories.bin")
        with BinaryDs(file, features=14) as dataset:
            self.assertEqual(dataset.get_categories(), 0)
            dataset.write(self.data_raw[:1])
        with BinaryDs(file, features=14) as dataset:
            self.assertEqual(dataset.get_categories(), 1)
            dataset.write(self.data_raw[2:3])
        with BinaryDs(file, features=14) as dataset:
            self.assertEqual(dataset.get_categories(), 3)



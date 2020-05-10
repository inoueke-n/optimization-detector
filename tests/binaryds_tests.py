import os
import shutil
import tempfile
from typing import List
from unittest import TestCase

from src.binaryds import BinaryDs

PREFIX = "BCCFLT_"


class TestBinaryDs(TestCase):
    tmpdir: str = None
    data_fun: List[bytes] = [
        b"\xFF",
        b"\xC0\x9C\x2E\x21\xFC\xC7\xDC\x94\x95\x35\xEA",
        b"\x89\x5E\xCB\xFB\x78\xA2\x78\x46\x16\x3D\xD7\xC6\xA4\x62\xC2\xE1\x3E"
        b"\x5F\xAD\xC4\x8E"
    ]
    data_raw: List[bytes] = [
        b"\x93\xE7\x32\x89\x1B\x08\x01\x65\x31\xA6\x64\x18\x3B\xA2\xA1\x32",
        b"\xDE\xDE\xF8\x1B\x83\x0E\xBF\xD0\x59\x04\x7C\x76\x33\x6F\xBE\xDE",
        b"\xB5\xAF\x4D\x2B\xBD\xDA\x32\x65\x61\x62\x90\x35\xEA\x7E\x5C\xAC",
    ]
    data_raw2: List[bytes] = [
        b"\xC0\xBB\x3B\x1E\x55\xB7\x45\x17\xC2\x83\x86\x33\xCE\xAB\x4C\x96",
        b"\xA7\x11\x87\xB2\x7D\xDF\x6E\x03\x13\x2F\xEA\x64\xCD\x3C\xBD\x59",
        b"\x2B\x30\x54\x22\xB3\xE2\xEA\x75\xC2\xF5\x2E\x74\xDB\xEC\x88\x69",
        b"\xC1\xC9\x9B\x88\x33\x48\x20\x4B\xC6\x1B\x38\xDF\x9E\xBD\xB8\x45",
        b"\x83\x52\xBE\x52\xC1\x5B\xB9\xD4\x1E\xFD\x2C\xA4\x0D\x63\x1F\x7C",
        b"\xD5\x33\x59\xC9\x62\x71\x48\x00\x62\x7A\x0D\x89\x3C\x73\x97\xB2",
        b"\xAF\xDA\x83\x9E\x46\x46\x79\xBA\x20\x1D\x56\x38\x98\x8D\x81\x4C",
        b"\xAA\xDE\xC4\xE3\x27\x70\x52\x84\x4C\xBB\x3B\x57\x7C\x34\x0B\x13"
    ]

    @classmethod
    def setUpClass(self):
        systmpdir = tempfile.gettempdir()
        self.tmpdir = tempfile.mkdtemp(prefix=PREFIX, dir=systmpdir)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.tmpdir)

    # write empty file and asserts no exception is thrown
    def test_create_empty(self):
        file = os.path.join(self.tmpdir, "create_empty.bin")
        binary = BinaryDs(file)
        with self.assertRaises(FileNotFoundError):
            binary.read()

    # try to create an empty file by overwriting an existing one. Assert that
    # IOError is raised and the original file is unmodified
    def test_create_existing_different_type(self):
        file = os.path.join(self.tmpdir, "create_existing_different_type.bin")
        original_data = b'Just some random data'
        with open(file, "wb") as fp:
            fp.write(original_data)
        with self.assertRaises(IOError):
            binary = BinaryDs(file)
            binary.read()
            binary.write()
        with open(file, "rb") as fp:
            buffer = fp.read()
            self.assertEqual(buffer, original_data)

    # add some raw data with the wrong number of features. Assert error
    def test_wrong_number_features_raw(self):
        file = os.path.join(self.tmpdir, "wrong_number_features_raw.bin")
        binary = BinaryDs(file)
        with self.assertRaises(AssertionError):
            binary.set(0, self.data_raw)

    # add some raw data with the wrong number of features.
    # Assert no error only if expected features > features input
    def test_wrong_number_features_fun(self):
        file = os.path.join(self.tmpdir, "wrong_number_features_fun.bin")
        binary = BinaryDs(file)
        binary.set_function_granularity(True)
        binary.set_features(128)
        try:
            binary.set(0, self.data_raw)
        except AssertionError:
            self.fail(
                "test_wrong_number_features_fun should not raise AssertionError")
        binary.set_features(8)
        with self.assertRaises(AssertionError):
            binary.set(0, self.data_raw)

    # load some data and change feature number. Assert that the previously
    # loaded data is discarded
    def test_change_feature_number(self):
        file = os.path.join(self.tmpdir, "change_feature_number.bin")
        binary = BinaryDs(file)
        binary.set_features(16)
        binary.set(0, self.data_raw)
        binary.set_features(8)
        new_data = binary.get(0)
        self.assertEqual(len(new_data), 0)

    # assert that for empty objects the minimum number of examples is zero.
    # then load some data and assert again the correct number
    def test_minimum_number_samples(self):
        file = os.path.join(self.tmpdir, "change_feature_number.bin")
        binary = BinaryDs(file)
        self.assertEqual(binary.min_examples(), 0)
        binary.set_features(16)
        binary.set(0, self.data_raw)
        binary.set(2, self.data_raw2)
        self.assertEqual(binary.min_examples(), 3)

    # Write a function granularity file. Then read it. Assert the content is ok
    def test_read_write_fun(self):
        file = os.path.join(self.tmpdir, "read_write_fun.bin")
        binary = BinaryDs(file)
        binary.set_function_granularity(True)
        binary.set_features(32)
        binary.set(0, self.data_raw)
        binary.set(2, self.data_fun)
        binary.write()
        binary = None
        binary = BinaryDs(file)
        self.assertNotEqual(binary.get_features(), 32)
        self.assertFalse(binary.get_function_granularity())
        self.assertEqual(binary.get_categories(), 0)
        self.assertEqual(binary.get(0), [])
        binary.read()
        self.assertEqual(binary.get_features(), 32)
        self.assertEqual(binary.get_categories(), 3)
        self.assertTrue(binary.get_function_granularity())
        self.assertEqual(binary.get(0), self.data_raw)
        self.assertEqual(binary.get(1), [])
        self.assertEqual(binary.get(2), self.data_fun)

    # Write a raw granularity file. Then read it. Assert the content is ok
    def test_read_write_raw(self):
        file = os.path.join(self.tmpdir, "read_write_raw.bin")
        binary = BinaryDs(file)
        binary.set_features(16)
        binary.set(0, self.data_raw)
        binary.set(3, self.data_raw2)
        binary.write()
        binary = None
        binary = BinaryDs(file)
        self.assertNotEqual(binary.get_features(), 16)
        self.assertFalse(binary.get_function_granularity())
        self.assertEqual(binary.get_categories(), 0)
        self.assertEqual(binary.get(0), [])
        binary.read()
        self.assertEqual(binary.get_features(), 16)
        self.assertEqual(binary.get_categories(), 4)
        self.assertFalse(binary.get_function_granularity())
        self.assertEqual(binary.get(0), self.data_raw)
        self.assertEqual(binary.get(1), [])
        self.assertEqual(binary.get(2), [])
        self.assertEqual(binary.get(3), self.data_raw2)

    # add unbalanced classes. Assert their samples number are not the same.
    # call rebalance and assert that the samples are now correct and the extra
    # ones are discarded
    def test_rebalance_no_tests(self):
        file_train = os.path.join(self.tmpdir, "rebalance.bin")
        train = BinaryDs(file_train)
        train.set_features(16)
        train.set(0, self.data_raw)
        train.set(1, self.data_raw2)
        self.assertNotEqual(len(train.get(0)), len(train.get(1)))
        train.rebalance(None)
        self.assertEqual(len(train.get(0)), len(train.get(1)))

    # add unbalanced classes. Assert their samples number are not the same.
    # call rebalance and assert that the samples are correct and the extra ones
    # are put in the test dataset
    def test_rebalance_with_tests(self):
        file_train = os.path.join(self.tmpdir, "rebalance_train.bin")
        file_test = os.path.join(self.tmpdir, "rebalance_test.bin")
        train = BinaryDs(file_train)
        test = BinaryDs(file_test)
        train.set_features(16)
        test.set_features(16)
        train.set(0, self.data_raw)
        train.set(1, self.data_raw2)
        test.set(0, [b"\xfd\xd7\x62\x42\x2a\x84\x63\xc4\x91\x44\xac\xb0\x24\x4b\x06\x8d"])
        test.set(1, [b"\x15\x66\x41\xfe\xa4\x92\x7a\xed\x7a\xdf\xab\x8d\xc8\x91\x03\xb0"])
        self.assertNotEqual(len(train.get(0)), len(train.get(1)))
        train.rebalance(test)
        self.assertEqual(len(train.get(0)), len(train.get(1)))
        self.assertEqual(len(test.get(0)), 1)
        self.assertEqual(len(test.get(1)), 6)

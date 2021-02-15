import os
from typing import List

from src.binaryds import BinaryDs


def run_summary(model_dir: str) -> None:
    """
    Gets a summary of the dataset contained in a directory
    :param model_dir: Path to the folder where the train.bin, test.bin and
    validate.bin can be found
    """
    assert (os.path.exists(model_dir))
    train_bin = os.path.join(model_dir, "train.bin")
    test_bin = os.path.join(model_dir, "test.bin")
    validate_bin = os.path.join(model_dir, "validate.bin")
    assert os.path.exists(train_bin), "Train dataset does not exists!"
    assert os.path.exists(test_bin), "Test dataset does not exists!"
    assert os.path.exists(validate_bin), "Validation dataset does not exists!"
    train = BinaryDs(train_bin, read_only=True).open()
    train_categories = count_categories(train)
    openc = not train.is_encoded()
    features = train.get_features()
    train.close()
    val = BinaryDs(validate_bin, read_only=True).open()
    val_categories = count_categories(val)
    val.close()
    test = BinaryDs(test_bin, read_only=True).open()
    test_categories = count_categories(test)
    test.close()
    print(f"Features: {features}")
    print(f"Number of classes: {len(train_categories)}")
    if openc:
        print("Type: opcode encoded")
    else:
        print("Type: raw values")
    print("--------------------")
    for i in range(0, len(train_categories)):
        print(f"Training examples for class {i}: {train_categories[i]}")
    for i in range(0, len(val_categories)):
        print(f"Validation examples for class {i}: {val_categories[i]}")
    for i in range(0, len(test_categories)):
        print(f"Testing examples for class {i}: {test_categories[i]}")


def count_categories(dataset: BinaryDs) -> List[int]:
    examples = dataset.get_examples_no()
    amount = 1000
    read_total = int(examples / amount)
    remainder = examples % amount
    categories = []
    for i in range(read_total):
        buffer = dataset.read(i * amount, amount)
        for val in buffer:
            category = val[0]
            while len(categories) <= category:
                categories.append(0)
            categories[category] += 1
    if remainder > 0:
        buffer = dataset.read(read_total * amount, remainder)
        for val in buffer:
            category = val[0]
            while len(categories) <= category:
                categories.append(0)
            categories[category] += 1
    assert len(categories) == dataset.get_categories()
    return categories

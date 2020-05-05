import argparse
import functools
import sys

from preprocess import run_preprocess

DIR = "/mnt/md0/flag_detection"
FEATURES = 2048


class FlagDetectionTrainer():
    def __init__(self):
        actions = ["preprocess", "train", "evaluate"]
        actions_desc = functools.reduce(lambda a, b: a + "\n\t" + b, actions)
        parser = argparse.ArgumentParser(
            description="Train a compiler and optimization detector",
            usage=f"{sys.argv[0]} <action> [<args>]\n"
                  f"Possible actions "
                  f"are:\n\t{actions_desc}",
            add_help=False)
        parser.add_argument("action",
                            choices=["preprocess", "merge", "train",
                                     "evaluate"],
                            help="The action that should be performed.")
        args = parser.parse_args(sys.argv[1:2])
        # dispatch function with same name of the action
        getattr(self, args.action)(sys.argv[2:])

    def preprocess(self, args):
        parser = argparse.ArgumentParser(
            description="Creates the train and test dataset from the "
                        "existing data",
            usage=f"{sys.argv[0]} preprocess [-h] -i data_dir -c class -m "
                  f"model_dir\n")
        parser.add_argument("-i", "--input", required=True,
                            metavar="data_dir",
                            help="path to the folder containing the "
                                 "unprocessed data")
        parser.add_argument("-c", "--classes", required=True, metavar="string",
                            help="a string of 0s and 1s representing the "
                                 "output class for this data")
        parser.add_argument("-m", "--model", metavar="model_dir",
                            required=True,
                            help="path to the folder that will contain the "
                                 "model. If an existing dataset is found, "
                                 "it will be merged with this one")
        parsed_args = parser.parse_args(args)
        run_preprocess(parsed_args.input, parsed_args.classes,
                       parsed_args.model)

    def train(self):
        pass

    def evaluate(self):
        pass


if __name__ == "__main__":
    FlagDetectionTrainer()

    # execute only if run as a script
    # function_grained = False
    # MODEL_DIR = "/mnt/md0/flag_detection/models/opt_clang_whole/"
    # write_dataset(MODEL_DIR, function_grained)
    # train = read_binary(MODEL_DIR + "train.bin", function_grained)
    # test = read_binary(MODEL_DIR + "test.bin", function_grained)
    # (X_train, y_train) = generate_sequences(train)
    # (X_test, y_test) = generate_sequences(test)
    ###########################################################################
    # # FUNCTION GRAINED
    # binary_plain_LSTM(X_train, y_train, X_test, y_test,
    #                   model_path=MODEL_DIR + "model.hdf5",
    #                   embedding_size=65536, pad_length=1024)
    # binary_convolutional_LSTM(X_train, y_train, X_test, y_test,
    #                           model_path=MODEL_DIR + "model.hdf5",
    #                   embedding_size=65536, pad_length=FEATURES)
    ###########################################################################
    # # EXECUTABLE GRAINED
    # binary_convolutional_LSTM(X_train, y_train, X_test, y_test,
    #                           model_path=MODEL_DIR + "model.hdf5",
    #                   embedding_size=256, pad_length=FEATURES)
    ###########################################################################
    # EVALUATION
    # cut = 400
    # while cut < 1100:
    #     print(f"Evaluating {cut}")
    #     test = read_binary(MODEL_DIR + "test.bin", function_grained)
    #     new_test = []
    #     for sample in test:
    #         if len(sample["x"]) >= cut:
    #             sample["x"] = sample["x"][:cut]
    #             new_test.append(sample)
    #     (X_test, y_test) = generate_sequences(test)
    #     X_test = X_test[:10000]
    #     y_test = y_test[:10000]
    #     matrix = evaluate_nn(MODEL_DIR + "model.hdf5", X_test, y_test,
    #                          pad_length=FEATURES)
    #     matrix = np.asarray(matrix)
    #     with open(
    #             "/mnt/md0/flag_detection/models/confusion_opt_clang_whole.csv",
    #             "a") as f:
    #         f.write(str(cut) + ",")
    #         f.write(str(matrix[0][0]) + ",")
    #         f.write(str(matrix[0][1]) + ",")
    #         f.write(str(matrix[1][0]) + ",")
    #         f.write(str(matrix[1][1]) + "\n")
    #     if cut < 25:  # more accurate evaluation where required
    #         cut = cut + 1
    #     elif cut < 80:
    #         cut = cut + 5
    #     else:
    #         cut = cut + 100

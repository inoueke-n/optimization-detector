import argparse
import functools
import multiprocessing
import sys

from src.evaluation import run_evaluation
from src.extractor import run_extractor
from src.preprocess import run_preprocess
from src.summary import run_summary
from src.train import run_train


class FlagDetectionTrainer:

    def __init__(self):
        actions = ["extract", "preprocess", "summary",
                   "train", "tune", "evaluate", "infer"]
        actions_desc = functools.reduce(lambda a, b: a + "\n\t" + b, actions)
        parser = argparse.ArgumentParser(
            description="Train a compiler and optimization detector",
            usage=f"{sys.argv[0]} <action> [<args>]\n"
                  f"Possible actions "
                  f"are:\n\t{actions_desc}",
            add_help=False)
        parser.add_argument("action",
                            choices=actions,
                            help="The action that should be performed.")
        args = parser.parse_args(sys.argv[1:2])
        # dispatch function with same name of the action
        getattr(self, args.action)(sys.argv[2:])

    @staticmethod
    def extract(args):
        parser = argparse.ArgumentParser(
            description="Extracts the unprocessed data from a binary file.",
            usage=f"{sys.argv[0]} extract [optional arguments] file "
                  f"output_dir\n"
        )
        parser.add_argument("input",
                            nargs="+",
                            metavar="file",
                            help="Binary file(s) that should be used for data "
                                 "extraction.")
        parser.add_argument("output_dir",
                            help="Folder that will be used for writing the "
                                 "extracted data.")
        parser.add_argument("-e", "--encoded", action="store_true",
                            help="Assumes opcode encoded analysis if set.")
        parser.add_argument("-j", "--jobs", required=False,
                            default=multiprocessing.cpu_count(),
                            help="Specifies the number of concurrent jobs. "
                                 "Default to the number of CPUs in the "
                                 "system.")
        parsed_args = parser.parse_args(args)
        run_extractor(parsed_args.input, parsed_args.output_dir,
                      parsed_args.encoded, parsed_args.jobs)

    @staticmethod
    def preprocess(args):
        parser = argparse.ArgumentParser(
            description="Creates the train and test dataset from the "
                        "existing data.",
            usage=f"{sys.argv[0]} preprocess [optional arguments] "
                  f"-c "
                  f"category data_dir output_dir\n")
        parser.add_argument("data_dir",
                            nargs="+",
                            help="Path to the folder(s) (or to single files) "
                                 "containing the unprocessed data")
        parser.add_argument("output_dir",
                            help="Path to the folder that will contain the "
                                 "preprocessed data. If an existing dataset is"
                                 " found, it will be merged with this one.")
        parser.add_argument("-e", "--encoded", action="store_true",
                            help="Assumes opcode encoded analysis if set.")
        parser.add_argument("-f", "--features", default=2048,
                            help="Number of features used in the evaluation, "
                                 "defaults to 2048.")
        parser.add_argument("-c", "--category", required=True, metavar="int",
                            help="A number representing the "
                                 "category label for this data.")
        parser.add_argument("-s", "--seed", default=None,
                            help="Seed used for the shuffling process")
        parser.add_argument("-b", "--balance", action="store_true",
                            help="Decides whether the amount of samples "
                                 "should be the same for every class or not.")
        parser.add_argument("--incomplete", action="store_true",
                            help="Generates an incomplete dataset, "
                                 "effectively skipping deduplication, "
                                 "shuffling and splitting between "
                                 "train/validation/test. This assumes that "
                                 "another preprocess will be called later "
                                 "without this flag enabled")
        parsed_args = parser.parse_args(args)
        run_preprocess(parsed_args.data_dir, int(parsed_args.category),
                       parsed_args.output_dir, parsed_args.encoded,
                       int(parsed_args.features), parsed_args.balance,
                       parsed_args.seed, parsed_args.incomplete)

    @staticmethod
    def train(args):
        parser = argparse.ArgumentParser(
            description="Train (or resume training) a model using the "
                        "previously generated data.",
            usage=f"{sys.argv[0]} train [optional args] model_dir\n")
        parser.add_argument("data_dir",
                            help="Folder for containing the test.bin and "
                                 "train.bin generated by the preprocess "
                                 "action. This train run will create a "
                                 "subfolder in this directory containing the "
                                 "trained model")
        parser.add_argument("-n", "--network",
                            default="cnn", choices=["dense", "lstm", "cnn"],
                            help="Choose which network to use for training.")
        parser.add_argument("-b", "--batchsize",
                            default=256, type=int)
        parser.add_argument("-s", "--seed", metavar="seed", default=0,
                            help="Seed used to initialize the weights during "
                                 "training.")
        parsed_args = parser.parse_args(args)
        run_train(parsed_args.data_dir, int(parsed_args.seed),
                  parsed_args.network, parsed_args.batchsize)

    @staticmethod
    def evaluate(args):
        parser = argparse.ArgumentParser(
            description="Run the evaluation on a trained model. This will "
                        "evaluate the accuracy with increasing "
                        "number of features. Additionally, with the parameter "
                        "--confusion it is possible to evaluate the confusion "
                        "matrix for a set number of features",
            usage=f"{sys.argv[0]} evaluate [optional args] data_dir\n")
        parser.add_argument("data_dir",
                            help="Folder for the model containing the "
                                 "test.bin generated by the preprocess action")
        parser.add_argument("-m", "--model_path", default="",
                            help="Trained model. Defaults to "
                                 "data_dir/model.h5")
        parser.add_argument("-o", "--output", metavar="filename",
                            default="output.csv",
                            help="Output file where the test results will be "
                                 "written. The file will be a .csv file.")
        parser.add_argument("-c", "--confusion", metavar="value", default=0,
                            help="Prints the confusion matrix for a single "
                                 "number of features.", type=int)
        parser.add_argument("-s", "--seed", metavar="seed", default="0",
                            help="Seed used to create the sequences.")
        parser.add_argument("-b", "--batchsize",
                            default=256, type=int)
        parsed_args = parser.parse_args(args)
        run_evaluation(parsed_args.data_dir, parsed_args.model_path,
                       parsed_args.output, int(parsed_args.seed),
                       parsed_args.confusion, parsed_args.batchsize)

    @staticmethod
    def summary(args):
        parser = argparse.ArgumentParser(
            description="Prints a summary of the preprocessed dataset.",
            usage=f"{sys.argv[0]} summary [-h] model_dir\n")
        parser.add_argument("model_dir",
                            help="Folder for the model containing the "
                                 "files generated by the preprocess action.")
        parsed_args = parser.parse_args(args)
        run_summary(parsed_args.model_dir)

    @staticmethod
    def infer(args):
        parser = argparse.ArgumentParser(
            description="Use a trained model to infer the optimization level "
                        "of some binaries.",
            usage=f"{sys.argv[0]} infer [optional args] -m model "
                  f"input_file\n")
        parser.add_argument("input", nargs="*", default=None,
                            help="List of files that will be used for "
                                 "inference.")
        parser.add_argument("-m", "--model", metavar="filename",
                            required=True, help="Path to the .h5 file "
                                                "containing the trained model")
        parser.add_argument("-o", "--output", required=False,
                            help="Output path of the prediction.")
        parser.add_argument("-b", "--batch", required=False,
                            default="256",
                            help="Maximum batch size for the model.")
        parser.add_argument("-d", "--dir", required=False,
                            default=None,
                            help="Directory containing .bin files")
        parser.add_argument("-f", "--features", default=2048,
                            help="Number of features used in the training, "
                                 "defaults to 2048.")
        parser.add_argument("-e", "--encoded", action="store_true",
                            help="Assumes opcode encoded analysis if set.")
        parser.add_argument("-t", "--threads", required=False,
                            default=multiprocessing.cpu_count(),
                            help="Specifies the number of concurrent jobs. "
                                 "Default to the number of CPUs in the "
                                 "system.")
        parsed_args = parser.parse_args(args)
        # run_inference(parsed_args.input,
        #               parsed_args.dir,
        #               parsed_args.model,
        #               parsed_args.encoded,
        #               parsed_args.output,
        #               int(parsed_args.batch),
        #               int(parsed_args.features),
        #               int(parsed_args.threads))


if __name__ == "__main__":
    FlagDetectionTrainer()

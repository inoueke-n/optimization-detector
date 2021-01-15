import argparse
import multiprocessing
import os
import shutil
from argparse import Namespace


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def println_ok():
    print(color.BOLD + color.GREEN + "OK" + color.END)


def println_err():
    print(color.BOLD + color.RED + "ERR" + color.END)


def println_warn(text: str):
    print("[" + color.BOLD + color.YELLOW + "WARN" + color.END + f"]: {text}")


def println_info(text: str):
    print("[" + color.BOLD + color.CYAN + "INFO" + color.END + f"]: {text}")


def getopt() -> Namespace:
    parser = argparse.ArgumentParser(
        description="Generates a dataset with the given "
                    "architecture/compiler, optimization level, and output "
                    "directory. ")
    parser.add_argument("output", default=None,
                        help="Folder where the final output will be placed")
    parser.add_argument("-j", "--jobs", required=False,
                        default=multiprocessing.cpu_count(),
                        help="Specifies the number of concurrent jobs. "
                             "Defaults to the number of CPUs in the "
                             "system.")
    parser.add_argument("--cc", required=False, type=str,
                        help="Name of the C compiler to be used. If more "
                             "than one, separate the paths with :",
                        default="cc")
    parser.add_argument("--cxx", required=False, type=str,
                        help="Name of the CXX compiler to be used. If more "
                             "than one, separate the paths with :",
                        default="c++")
    parser.add_argument("-o", "--opts", type=str,
                        help="String with the optimization levels that will "
                             "be passed to each compilation. Should contain "
                             "only the numbers 0,1,2,3 or s separated by :",
                        default="0:1:2:3:s")
    return parser.parse_args()


def check_flags(args: Namespace) -> Namespace:
    print("Checking args... ", end="")
    info = []
    cc = list(filter(None, args.cc.split(":")))
    cxx = list(filter(None, args.cxx.split(":")))
    flags = list(filter(None, args.opts.split(":")))
    if len(cc) != len(cxx):
        println_err()
        print(f"--cc and --cxx variables should have the same length. Got "
              f"{cc} and {cxx} respectively")
        exit(1)
    if len(cc) == 0:
        cc = ['cc']
        cxx = ['c++']
        info.append("Compilers have been set to default values. (None were "
                    "passed)")
    allowed_flags = {"0", "1", "2", "3", "s"}
    for flag in flags:
        if flag not in allowed_flags:
            flags.remove(flag)
            info.append(f"Removed flag {flag}. Not supported")
    if len(flags) == 0:
        flags = list(allowed_flags)
        info.append("Optimization flags have been set to default values. ("
                    "None were passed)")
    wrong_cc = []
    which(cc, wrong_cc)
    which(cxx, wrong_cc)
    if len(wrong_cc) > 0:
        println_err()
        print(f"Could not find the following compilers: {wrong_cc}")
    println_ok()
    for msg in info:
        println_info(msg)
    args.cc = cc
    args.cxx = cxx
    args.flags = flags
    return args


def which(cc, wrong_cc):
    for i, exe in enumerate(cc):
        if not os.path.isfile(exe) or not os.access(exe, os.X_OK):
            abs = shutil.which(exe)
            if abs is not None:
                cc[i] = abs
            else:
                wrong_cc.append(exe)


def check_host_system():
    print("Checking host system... ", end="")
    set = {("bash", "bash"),
           ("bison", "bison"),
           ("bzip2", "bzip2"),
           ("chown", "coreutils"),
           ("diff", "diffutils"),
           ("find", "findutils"),
           ("gawk", "gawk"),
           ("grep", "grep"),
           ("gzip", "gzip"),
           ("m4", "m4"),
           ("make", "make"),
           ("patch", "patch"),
           ("perl", "perl"),
           ("python3", "python"),
           ("sed", "sed"),
           ("tar", "tar"),
           ("makeinfo", "texinfo"),
           ("xz", "xz"),
           }
    missing = []
    for program in set:
        if shutil.which(program[0]) is None:
            missing.append(program[1])
    if len(missing) > 0:
        println_err()
        print(f"Missing the following programs {missing}")
        exit(1)
    else:
        println_ok()


if __name__ == "__main__":
    args = getopt()
    args = check_flags(args)
    check_host_system()

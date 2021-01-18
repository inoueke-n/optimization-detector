import argparse
import hashlib
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import tarfile
import urllib.request
import urllib.error
from argparse import Namespace
from typing import Dict, List

from tqdm import tqdm


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
    """
    Prints OK in bold green
    """
    print(color.BOLD + color.GREEN + "OK" + color.END, flush=True)


def println_err():
    """
    Prints ERR in bold red
    """
    print(color.BOLD + color.RED + "ERR" + color.END, flush=True)


def println_warn(text: str):
    """
    Prints WARN in yellow, and the warning message
    """
    print("[" + color.BOLD + color.YELLOW + "WARN" + color.END + f"]: {text}",
          flush=True)


def println_info(text: str):
    """
    Prints INFO in cyan, and the info message
    """
    print("[" + color.BOLD + color.CYAN + "INFO" + color.END + f"]: {text}",
          flush=True)


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
    """
    Asserts that the parameters passed to the script are consistent and
    respects the specification. Moreover checks that the compilers exists
    and creates the output folder if not existing.
    :param args arguments received from ArgumentParser
    """
    print("Checking args... ", end="", flush=True)
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
    if not os.path.exists(args.output):
        try:
            os.makedirs(args.output)
            info.append("Created output directory")
        except IOError as err:
            println_err()
            print(f"Could not create directory {args.output}:")
            print(err)
            exit(1)
    elif not os.path.isdir(args.output) or not os.access(args.output, os.W_OK):
        println_err()
        print(f"Output directory {args.output} already exists and is not "
              f"writable")
        exit(1)
    println_ok()
    for msg in info:
        println_info(msg)
    args.cc = cc
    args.cxx = cxx
    args.flags = flags
    return args


def which(exe_list: List[str], wrong_exe: List[str]):
    """
    Locate some executables. Given a list of executable, locates them by
    looking in the path and replaces their name with the absolute path. If
    the executable can not be located, it is appended to the `wrong_exe` list
    """
    for i, exe in enumerate(exe_list):
        if not os.path.isfile(exe) or not os.access(exe, os.X_OK):
            abs = shutil.which(exe)
            if abs is not None:
                exe_list[i] = abs
            else:
                wrong_exe.append(exe)


def check_host_system():
    """
    Checks the presence of some programs required by the current script
    """
    print("Checking host system... ", end="", flush=True)
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


def check_hash(folder: str, md5s: Dict, warn: set):
    """
    Asserts that all files in the given folder have the MD5 hash passed in the
    dictionary as {"filename","hash"}. The last parameter, will be used to
    append warning messages
    """
    checked = set()
    for file in os.listdir(folder):
        filename = os.path.join(folder, file)
        hash = hashlib.md5()
        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash.update(chunk)
        hash = hash.hexdigest()
        if file in md5s:
            if md5s[file] == hash:
                checked.add(file)
            else:
                println_err()
                print(f"Wrong hash for downloaded file {file} (Expected "
                      f"{md5s[file]}, got {hash})")
                exit(1)
        else:
            warn.add(f"Foreign file in sources folder: {file}")
    difference = set(md5s.keys()) - checked
    if len(difference) > 0:
        println_err()
        print(f"Missing the following downloads (listed in the md5 "
              f"files): {difference}")
        exit(1)
    else:
        println_ok()


def prepare_folder(args: Namespace) -> Namespace:
    """
    Downloads required files and compress them into an archive.
    Then extracts the archive in a temporary folder. Returns the same
    arguments as input, but with the extra temporary folder.
    """
    warn = set()
    md5s = {"": ""}  # these avoids problems with newlines
    with open("resources/md5sums", "r") as fp:
        for line in fp.read().splitlines():
            split = list(filter(None, line.split(" ")))
            md5s[split[1]] = split[0]
    md5s.pop("")
    if not os.path.exists("resources/sources.tar"):
        print("Retrieving software... ", end="", flush=True)
        urls = {"": ""}
        with open("resources/wget-list", "r") as fp:
            for url in fp.read().splitlines():
                urls[url.rsplit('/', 1)[-1]] = url
        with open("resources/md5sums", "r") as fp:
            for line in fp.read().splitlines():
                split = list(filter(None, line.split(" ")))
                md5s[split[1]] = split[0]
        urls.pop("")
        if len(urls) != len(md5s):
            println_err()
            print("Different numbers of downloads and md5s in resources")
            exit(1)
        else:
            print("")
        for key in tqdm(urls.keys(), file=sys.stdout):
            try:
                filename = os.path.join("resources/sources", key)
                urllib.request.urlretrieve(urls[key], filename)
            except urllib.error.HTTPError as e:
                print(f"Failed to retrieve file {e.url}: {e.code} {e.msg}")
                exit(1)
        print("Checking downloaded files... ", end="", flush=True)
        check_hash("resources/sources", md5s, warn)
        print("Compressing... ", end="", flush=True)
        # no compression, as this is an archive of archives
        with tarfile.open("resources/sources.tar", "w") as tar:
            tar.add("resources/sources", arcname="sources")
        # delete original files, at this point they are not needed anymore
        for file in os.listdir("resources/sources"):
            if file in urls:
                os.remove(os.path.join("resources/sources", file))
        println_ok()
    args.build_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(args.build_dir, "logs"))
    print("Decompressing... ", end="", flush=True)
    with tarfile.open("resources/sources.tar", "r") as tar:
        tar.extractall(args.build_dir)
    println_ok()
    print("Checking extracted files... ", end="", flush=True)
    check_hash(args.build_dir + "/sources", md5s, warn)
    for msg in warn:
        println_warn(msg)
    return args


def build(args: Namespace):
    """
    Runs the various bash scripts that will build the dataset
    """
    myenv = os.environ.copy()
    myenv["CC"] = args.cc[0]
    myenv["CXX"] = args.cxx[0]
    myenv["CFLAGS"] = "-O0"
    myenv["CXXFLAGS"] = myenv["CFLAGS"]
    src_dir = os.path.join(args.build_dir, "sources")
    err = []
    print("Building...")
    for script in tqdm(sorted(os.listdir("resources/scripts")),
                       file=sys.stdout):
        script_abs = os.path.abspath(os.path.join("resources/scripts", script))
        outfile = os.path.join(args.build_dir, "logs")
        errfile = os.path.join(outfile, script + ".stderr.log")
        outfile = os.path.join(outfile, script + ".stdout.log")
        script_args = ["bash", script_abs, args.build_dir, str(args.jobs)]
        outfile = open(outfile, "w")
        errfile = open(errfile, "w")
        process = subprocess.Popen(script_args, env=myenv, cwd=src_dir,
                                   stdout=outfile, stderr=errfile)
        process.wait()
        outfile.close()
        errlen = errfile.tell()
        errfile.close()
        if errlen > 0:
            err.append(script)
    if len(err):
        println_warn(f"The following scripts did not run successfully: {err}")



if __name__ == "__main__":
    args = getopt()
    args = check_flags(args)
    check_host_system()
    args = prepare_folder(args)
    build(args)
    shutil.rmtree(args.build_dir)

import argparse
import fnmatch
import hashlib
import itertools
import lzma
import multiprocessing
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from argparse import Namespace
from typing import Dict, List, Tuple

import magic
from tqdm import tqdm


class Color:
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
    print(Color.BOLD + Color.GREEN + "OK" + Color.END, flush=True)


def println_err():
    """
    Prints ERR in bold red
    """
    print(Color.BOLD + Color.RED + "ERR" + Color.END, flush=True)


def println_warn(text: str):
    """
    Prints WARN in yellow, and the warning message
    """
    print("[" + Color.BOLD + Color.YELLOW + "WARN" + Color.END + f"]: {text}",
          flush=True)


def println_info(text: str):
    """
    Prints INFO in cyan, and the info message
    """
    print("[" + Color.BOLD + Color.CYAN + "INFO" + Color.END + f"]: {text}",
          flush=True)


def getopt() -> Namespace:
    """
    Parses the command line arguments
    """
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
    parser.add_argument("-t", "--targets", required=False, type=str,
                        help="Toolchain triplet. A triplet like "
                             "x86_64-linux-gnu is expected. This will be "
                             "used for the target platform. The complete "
                             "toolchain is expected to be located in "
                             "/usr/<toolchain triplet>. If more "
                             "than one, separate the paths with :",
                        default="cc")
    parser.add_argument("-o", "--opts", type=str,
                        help="String with the optimization levels that will "
                             "be passed to each compilation. Should contain "
                             "only the numbers 0,1,2,3 or s separated by :",
                        default="0:1:2:3:s")
    parser.add_argument("-l", "--logdir", type=str,
                        help="Directory that will contain the building logs "
                             "for each binary. These will be packed in a "
                             "single archive. By default, no logs will be "
                             "saved", default=None)
    return parser.parse_args()


def check_and_create_dir(dirpath: str, info: List[str], name: str):
    """
    Checks if a directory exists and is writable. Creates it if it does not 
    exist.
    :param dirpath: directory path 
    :param info: list of info messages
    :param name: informal name of the folder (like output for output folder or 
    log for log folder)
    """
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
            info.append(f"Created {name} directory")
        except IOError as err:
            println_err()
            print(f"Could not create directory {dirpath}:")
            print(err)
            exit(1)
    elif not os.path.isdir(dirpath) or not os.access(dirpath, os.W_OK):
        println_err()
        print(
            f"{name.capitalize()} directory {dirpath} already exists and is not "
            f"writable")
        exit(1)


def check_flags(args: Namespace) -> Namespace:
    """
    Asserts that the parameters passed to the script are consistent and
    respects the specification. Moreover checks that the compilers exists
    and creates the output folder if not existing.
    :param args arguments received from getopt()
    """

    info = []
    triplets = list(filter(None, args.targets.split(":")))
    flags = list(filter(None, args.opts.split(":")))
    print(f"Building for the following targets: {triplets}")
    print(f"Building following flags: {flags}")
    print("Checking args... ", end="", flush=True)
    allowed_flags = {"0", "1", "2", "3", "s"}
    for flag in flags:
        if flag not in allowed_flags:
            flags.remove(flag)
            info.append(f"Removed flag {flag}. Not supported")
    if len(flags) == 0:
        flags = list(allowed_flags)
        info.append("Optimization flags have been set to default values. ("
                    "None were passed)")
    check_and_create_dir(args.output, info, "output")
    if args.logdir is not None:
        check_and_create_dir(args.logdir, info, "logs")
    println_ok()
    for msg in info:
        println_info(msg)
    args.targets = triplets
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


def missing_toolchain(triplet: str) -> bool:
    """
    Checks whether gcc, g++ and binutils are installed and in the path for the
    current triplet
    :param triplet: a triplet in the form riscv64-linux-gnu
    :return: True if some part of the toolchain is missing, False otherwise
    """
    toolchain_expected = {"ar", "as", "gcc", "g++", "ld", "ranlib", "strip"}
    retval = False
    for tool in toolchain_expected:
        retval |= shutil.which(cmd=triplet + "-" + tool, mode=os.X_OK) is None
    return retval


def check_host_system(args: Namespace):
    """
    Checks the presence of some programs required by the current script
    """
    print("Checking host system... ", end="", flush=True)
    set = {("bash", "bash"),
           ("autoreconf", "autoconf"),
           ("automake", "automake"),
           ("bison", "bison"),
           ("bzip2", "bzip2"),
           ("chown", "coreutils"),
           ("cmake", "cmake"),
           ("diff", "diffutils"),
           ("file", "file"),
           ("find", "findutils"),
           ("flex", "flex"),
           ("gcc", "gcc"),
           ("g++", "g++"),
           ("gawk", "gawk"),
           ("gettext", "gettext"),
           ("grep", "grep"),
           ("groff", "groff"),
           ("gzip", "gzip"),
           ("libtoolize", "libtool"),
           ("m4", "m4"),
           ("make", "make"),
           ("meson", "meson"),
           ("ninja", "ninja-build"),
           ("patch", "patch"),
           ("perl", "perl"),
           ("python3", "python3"),
           ("sed", "sed"),
           ("tar", "tar"),
           ("makeinfo", "texinfo"),
           ("xz", "xz")
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
    print("Checking compilers... ", end="", flush=True)
    missing = list(filter(missing_toolchain, args.targets))
    uid = os.getuid()
    warn = ""
    if len(missing) != 0:
        if uid == 0:  # user is root
            for miss in missing:  # install the missing toolchains
                args = ["apt-get", "install", "-y", "binutils-" + miss,
                        "gcc-" + miss, "g++-" + miss]
                proc = subprocess.Popen(args, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                proc.wait()
                if proc.returncode == 0:
                    warn += f"{Color.BOLD}{miss}{Color.END} "
                else:
                    println_err()
                    msg = f"Could not install toolchain " \
                          f"{Color.BOLD}{miss}{Color.END}"
                    print(msg)
                    exit(1)
        else:
            println_err()
            msg = ""
            for miss in missing:
                msg += Color.BOLD + miss + Color.END + ", "
            msg = msg[:-2] + " toolchain(s) missing"
            print(msg)
            exit(1)
    println_ok()
    if warn != "":
        println_info(f"Installed the following toolchain(s): {warn}.")


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
        os.makedirs("resources/sources", exist_ok=True)
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
        for key in tqdm(urls.keys(), file=sys.stdout, ncols=60):
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
    print("Decompressing... ", end="", flush=True)
    with tarfile.open("resources/sources.tar", "r") as tar:
        tar.extractall(args.build_dir)
    println_ok()
    print("Checking extracted files... ", end="", flush=True)
    check_hash(args.build_dir + "/sources", md5s, warn)
    for msg in warn:
        println_warn(msg)
    return args


def clean_dir(prefix: str):
    """
    Removes all the files inside the prefix directory, except sources
    """
    for entry in os.listdir(prefix):
        if entry != "sources":
            filepath = os.path.join(prefix, entry)
            if os.path.isdir(filepath):
                shutil.rmtree(os.path.join(prefix, entry))
            else:
                os.remove(filepath)


def build_all(args: Namespace):
    """
    Runs the various bash scripts that will build the dataset
    """
    for triplet in args.targets:
        for opt in args.flags:
            clean_dir(args.build_dir)
            os.makedirs(os.path.join(args.build_dir, "logs"))
            build_single(args.build_dir, triplet, opt, args.jobs)
            strip(args.build_dir, triplet + "-strip")
            name = triplet.split("-")[0] + "-gcc-o" + opt
            pack_binaries(args.build_dir, name, args.output)
            if args.logdir is not None:
                pack_logs(args.build_dir, name, args.logdir)


def set_build_tools(triplet: str) -> Dict:
    """
    Sets the environment for compilation. This will actually set
    TOOL=triplet+tool where tool can be gcc, g++, etc...
    :param triplet: a triplet in the form "riscv64-linux-gnu"
    :return a dictionary corresponding to the environment
    """
    env = os.environ.copy()
    env["CC"] = triplet + "-gcc"
    env["CXX"] = triplet + "-g++"
    env["LD"] = triplet + "-ld"
    env["AR"] = triplet + "-ar"
    env["AS"] = triplet + "-as"
    env["STRIP"] = triplet + "-strip"
    env["RANLIB"] = triplet + "-ranlib"
    return env


def create_toolchain_cmake(sysroot: str, triplet: str):
    """
    Creates the toolchain.cmake file used by cmake scripts to cross-compile
    :param sysroot: the sysroot of the toolchain
    :param triplet: a triplet in the form riscv64-linux-gnu
    """
    filename = os.path.join(sysroot, "toolchain.cmake")
    with open(filename, "w") as fp:
        fp.write("set(CMAKE_SYSTEM_NAME Linux)\n")
        fp.write(f"set(CMAKE_SYSTEM_PROCESSOR {triplet.split('-')[0]})\n")
        fp.write(f"set(CMAKE_SYSROOT {sysroot})\n")
        fp.write(f"set(CMAKE_STAGING_PREFIX {sysroot})\n")
        fp.write(f"set(CMAKE_C_COMPILER {triplet}-gcc)\n")
        fp.write(f"set(CMAKE_CXX_COMPILER {triplet}-g++)\n")
        fp.write("set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)\n")
        fp.write("set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)\n")
        fp.write("set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)\n")
        fp.write("set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)\n")


def build_single(prefix: str, triplet: str, opt: str, jobs: int):
    """
    Builds all the software for a single architecture/triplet and optimization
    level.
    :param prefix: folder where the files will be placed
    :param triplet: a triplet in the form riscv64-linux-gnu
    :param opt: optimization level: 0, 1, 2, 3 or s
    :param jobs: number of maximum concurrent jobs
    """
    src_dir = os.path.join(prefix, "sources")
    myenv = set_build_tools(triplet)
    create_toolchain_cmake(prefix, triplet)
    myenv["CFLAGS"] = " -pipe -O" + opt
    myenv["PKG_CONFIG_PATH"] = os.path.join(prefix, "usr/lib/pkgconfig")
    myenv["CXXFLAGS"] = myenv["CFLAGS"]
    print(f"Building {Color.BOLD}{triplet}{Color.END} with "
          f"optimization {Color.BOLD}-O{opt}{Color.END}...")
    script_list = sorted(os.listdir("resources/scripts"))
    pbar = tqdm(script_list, file=sys.stdout, ncols=79)
    pbar.set_description("Building")
    for script in pbar:
        pbar.set_postfix_str(f"{Color.BOLD}{script.split('-')[1]}{Color.END}")
        script_abs = os.path.abspath(
            os.path.join("resources/scripts", script))
        outfile = os.path.join(prefix, "logs")
        outfile = os.path.join(outfile, script + ".stdout.log")
        script_args = ["bash", script_abs, prefix, str(jobs), opt, triplet]
        outfile = open(outfile, "w")
        process = subprocess.Popen(script_args, env=myenv, cwd=src_dir,
                                   stdout=outfile, stderr=subprocess.STDOUT)
        process.wait()
        outfile.close()


def get_bin_and_libs(prefix: str) -> Tuple[List[str], List[str]]:
    """
    Returns the list of binaries and libraries in a given folder (recursively).
    :param prefix: The folder that will be scanned
    :return: A tuple containing the list of binaries and list of libraries
             respectively
    """
    lib = os.path.join(prefix, "lib")
    usrlib = os.path.join(prefix, "usr/lib")
    bin = os.path.join(prefix, "bin")
    sbin = os.path.join(prefix, "sbin")
    usrbin = os.path.join(prefix, "usr/bin")
    usrsbin = os.path.join(prefix, "usr/sbin")
    usrlibex = os.path.join(prefix, "usr/libexec")
    lib_iter = itertools.chain(os.walk(lib), os.walk(usrlib))
    bin_iter = itertools.chain(os.walk(bin), os.walk(sbin), os.walk(usrbin),
                               os.walk(usrsbin), os.walk(usrlibex))
    bins = []
    libs = []
    for path, _, files in bin_iter:
        for filename in files:
            fullpath = os.path.join(path, filename)
            if not os.path.islink(fullpath):
                bins.append(fullpath)
    for path, _, files in lib_iter:
        for filename in files:
            fullpath = os.path.join(path, filename)
            if not os.path.islink(fullpath):
                libs.append(os.path.join(path, filename))
    return bins, libs


def strip(prefix: str, strip: str):
    """
    Strips the files generated by the build_single function.
    :param prefix: the same folder passed to build_single
    :param strip: the strip program for the correct architecture
    """
    (bins, libs) = get_bin_and_libs(prefix)
    for lib in libs:
        if lib.endswith(".a"):
            subprocess.Popen([strip, "--strip-debug", lib],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        if fnmatch.fnmatch(lib, "*.so*"):
            subprocess.Popen([strip, "--strip-unneeded", lib],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    for bin in bins:
        subprocess.Popen([strip, "--strip-all", bin],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)


def pack_binaries(prefix: str, name: str, output: str):
    """
    Packs all binaries generated by the build_single function into a single
    .tar.xz with e9 compression level
    :param prefix: the same folder passed to the build_single function
    :param name: name of the output tar.xz (without extension)
    :param output: the folder where the tarball will be created
    """
    (bins, libs) = get_bin_and_libs(prefix)
    files = []
    for bin in itertools.chain(bins, libs):
        if os.path.exists(bin) and os.access(bin, os.R_OK):
            desc = magic.from_file(bin)
            if desc.startswith("ELF"):
                files.append(bin)
    target_folder = os.path.join(prefix, name)
    target_tar = os.path.join(output, name + ".tar.xz")
    if os.path.exists(target_tar):
        target_tar = target_tar + str(time.time())
        println_warn(f"Output file already existing. "
                     f"The current output is thus {target_tar}")
    os.mkdir(target_folder)
    for file in files:
        if os.path.exists(file) and os.access(file, os.R_OK):
            shutil.copy(file, target_folder)
    with tarfile.open(target_tar, "w:xz",
                      preset=9 | lzma.PRESET_EXTREME) as tar:
        tar.add(target_folder, arcname=name)
    shutil.rmtree(target_folder)


def pack_logs(prefix: str, name: str, output: str):
    """
    Packs all logs generated by the build_single function into a single .tar.xz
    with e9 compression level
    :param prefix: the same folder passed to the build_single function
    :param name: name of the output tar.xz (without extension)
    :param output: the folder where the tarball will be created
    """
    target_tar = os.path.join(output, name + "-logs.tar.xz")
    if os.path.exists(target_tar):
        target_tar = target_tar + str(time.time())
        println_warn(f"Output file already existing. "
                     f"The current output is thus {target_tar}")
    with tarfile.open(target_tar, "w:xz",
                      preset=9 | lzma.PRESET_EXTREME) as tar:
        tar.add(os.path.join(prefix, "logs"), arcname=name + "-logs")


if __name__ == "__main__":
    args = getopt()
    args = check_flags(args)
    check_host_system(args)
    args = prepare_folder(args)
    build_all(args)
    shutil.rmtree(args.build_dir)

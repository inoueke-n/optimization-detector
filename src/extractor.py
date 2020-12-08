import os
from typing import List

import r2pipe as r2pipe
from tqdm import tqdm


def run_extractor(input_files: List[str], outdir: str, function: bool) -> None:
    """
    Extracts the data from binary files, either as a list of function
    opcodes or just the raw .text section.
    :param input_files: A list of string, each string representing a path to a
    binary file.
    :param outdir: The directory where the extracted data should be written.
    The same filename of the input_files will be used, with a .txt appended
    in case of function analysis or .bin otherwise.
    :param function: true if function analysis is requested. This particular
    type of analysis uses the output of disassembly instead of plain raw bytes.
    """
    if os.path.exists(outdir):
        if os.path.isdir(outdir):
            if os.access(outdir, os.W_OK):
                pass
            else:
                raise IOError(f"Folder {outdir} is not writable")
        else:
            raise IOError(f"{outdir} is not a folder")
    else:
        raise IOError(f"The folder {outdir} does not exist")

    if function:
        extension = ".txt"
    else:
        extension = ".bin"
    for file in tqdm(input_files):
        name = os.path.basename(file)
        if function:
            pass
        else:
            extract_dot_text_to_file(file,
                                     os.path.join(outdir, name + extension))


def extract_dot_text_to_file(file: str, out_file: str):
    """
    Extracts the raw .text section from a binary file and saves it to another
    file.
    :param file: path to the input file.
    :param out_file: The file where the dump will be saved.
    """
    r2 = r2pipe.open(file, ["-2"])
    sections = r2.cmdj("iSj")
    for section in sections:
        if section["name"] == ".text":
            address = section["vaddr"]
            length = section["size"]
            r2.cmd("s " + str(address))
            r2.cmd("pr " + str(length) + " > " + out_file)
            break
    r2.quit()


def extract_dot_text(file: str) -> List[bytes]:
    """
    Extracts and returns the raw .text section from a binary file.
    :param file: path to the input file.
    :return A bytearray containing the extracted data as a sequence of bytes.
    """
    data = None
    r2 = r2pipe.open(file, ["-2"])
    sections = r2.cmdj("iSj")
    for section in sections:
        if section["name"] == ".text":
            address = section["vaddr"]
            length = section["size"]
            r2.cmd("s " + str(address))
            data = r2.cmdj("pxj " + str(length))
            break
    r2.quit()
    return data

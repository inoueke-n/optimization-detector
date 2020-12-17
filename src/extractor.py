import csv
import os
from typing import List

import r2pipe as r2pipe
from tqdm import tqdm

invalid_opcodes_table = [False, False, False, False, False, False, True, True,
                         False, False, False, False, False, False, True, False,
                         False, False, False, False, False, False, True, True,
                         False, False, False, False, False, False, True, True,
                         False, False, False, False, False, False, True, True,
                         False, False, False, False, False, False, True, True,
                         False, False, False, False, False, False, True, True,
                         False, False, False, False, False, False, True, True,
                         True, True, True, True, True, True, True, True, True,
                         True, True, True, True, True, True, True, False,
                         False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False,
                         False, True, True, True, False, True, True, True,
                         True, False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False,
                         False, False, False, False, False, True, False, False,
                         False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False, True,
                         True, False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False,
                         False, False, False, False, False, True, True, False,
                         False, False, False, False, False, False, False,
                         False, False, False, False, False, False, True, True,
                         True, False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False,
                         False, False, False, False, False, False]


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
        extension = ".csv"
    else:
        extension = ".bin"
    for file in tqdm(input_files):
        name = os.path.basename(file)
        outfile = os.path.join(outdir, name + extension)
        if function:
            extract_function_to_file(file, outfile)
        else:
            extract_dot_text_to_file(file, outfile)


def dot_text_name(r2: r2pipe) -> str:
    """
    Returns the name of the dot text section.
    The name of this session is slightly different for the various OSes.
    :param r2: Opened r2pipe, used to gather info about the binary file
    :return: A string with the .text name inside the binary.
    """
    info = r2.cmdj("ij")
    bint = info["bin"]["bintype"]
    if bint == "mach0":
        return "0.__TEXT.__text"
    elif bint == "elf" or bint == "pe":
        return ".text"
    else:
        raise ValueError(f"Unknown file format {bint}")


def extract_dot_text_to_file(file: str, out_file: str) -> None:
    """
    Extracts the raw .text section from a binary file and saves it to another
    file.
    :param file: path to the input file.
    :param out_file: The file where the dump will be saved.
    """
    r2 = r2pipe.open(file, ["-2"])
    sections = r2.cmdj("iSj")
    expected_name = dot_text_name(r2)
    for section in sections:
        if section["name"] == expected_name:
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
    expected_name = dot_text_name(r2)
    for section in sections:
        if section["name"] == expected_name:
            address = section["vaddr"]
            length = section["size"]
            r2.cmd("s " + str(address))
            data = r2.cmdj("pxj " + str(length))
            break
    r2.quit()
    return data


def get_opcode(bytes: bytearray) -> bytearray:
    """
    Extracts the opcode from a statement, discarding prefixes and suffixes
    :param bytes: a bytearray containing the input statement
    :return: a bytearray containing the output opcode
    """
    # TODO: add ARM support
    prev_0f = False
    # 0xF2 or 0xF3 if previous value was one of those
    # used to address the sequence 0xF20FXX where 0FXX is the opcode
    prev_f23 = 0x00
    for byte in bytes:
        if not prev_0f:
            if byte == 0x0F:
                prev_0f = True
                prev_f23 = 0x00
            elif byte == 0xF2 or byte == 0xF3:
                prev_f23 = byte
            elif not invalid_opcodes_table[byte]:
                if prev_f23 != 0x00:
                    # 0xF2 or 0xF3 was not followed by 0FXX
                    return bytearray([prev_f23])
                else:
                    return bytearray([byte])
            else:
                # prefix that should be discarded
                pass
        else:
            return bytearray([0x0F, byte])
    return bytearray([0xFF, 0xFF])


def extract_function_to_file(file: str, out_file: str) -> None:
    """
    Opens a file and extract every function opcodes. Saves the result in a .csv
    The output .csv contains a function for each row with the following fields:
    - virtual address offset (decimal) of the current function in the binary
    - function name
    - function length (in bytes)
    - function bytes
    - function bytes without prefixes or suffixes
    :param file: The input binary file
    :param out_file: The output csv containing the extracted data
    """
    r2 = r2pipe.open(file, ["-2"])
    r2.cmd("aaa")
    imports = r2.cmdj("iij")
    import_set = {imp["plt"] for imp in imports}
    functions = r2.cmdj("aflj")
    rows = []
    if functions is not None:  # some files contain 0 functions
        for function in functions:
            if function["offset"] not in import_set:
                r2.cmd(f"s {function['offset']}")
                func = r2.cmdj("pdrj")
                opcodes = bytearray()
                raw_opcodes = []
                for stmt in func:
                    if "bytes" in stmt:  # invalid opcodes do not have "bytes"
                        opcodes_arr = get_opcode(
                            bytearray.fromhex(stmt["bytes"]))
                        opcodes.extend(opcodes_arr)
                        raw_opcodes.append(stmt["bytes"])
                rows.append(
                    [function["offset"], function["name"], function["size"],
                     ''.join(x for x in raw_opcodes),
                     ''.join(format(x, '02x') for x in opcodes)])
    with open(out_file, "w") as fp:
        writer = csv.writer(fp, delimiter=",", quotechar='"',
                            quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["offset", "name", "size", "raw", "opcodes"])
        writer.writerows(rows)
    return

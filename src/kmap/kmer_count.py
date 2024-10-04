import numpy as np
from Bio import SeqIO
import gzip
import click
from typing import Callable, Dict, List, Tuple
import pickle
import taichi as ti
from taichi.algorithms import parallel_sort
from scipy.stats import norm

from pathlib import Path
from dataclasses import dataclass, fields
import tomllib
import tomli_w

from .taichi_core import (kmer2hash_kernel_uint32, kmer2hash_kernel_uint64,
                         revcom_hash_kernel_uint32, revcom_hash_kernel_uint64,
                         cal_ham_dist_kernel_uint32, cal_ham_dist_kernel_uint64,
                         cal_partial_ham_dist_head_kernel_uint32, cal_partial_ham_dist_head_kernel_uint64,
                         cal_partial_ham_dist_tail_kernel_uint32, cal_partial_ham_dist_tail_kernel_uint64)

import pandas as pd

from importlib.resources import files, as_file

FileNameDict = {
    "default_config_file": "default_config.toml",
    "config_file": "config.toml",
    "default_motif_def_file": "default_motif_def_table.csv",
    "motif_def_file": "motif_def_table.csv",
    "processed_fasta_file": "input.bin.pkl",
    "processed_fasta_seqboarder_file": "input.seqboarder.bin.pkl",
    "motif_pos_density_file": "motif_pos_density.np.pkl",
    "motif_pos_density_plot_dir": "motif_pos_density",
    "kmer_count_dir": "kmer_count",
    "conseq_similarity_dir": "conseq_similarity",
    "co_occur_dir": "co_occurence",
    "co_occur_dist_mat_file": "co_occurence_motif_dist_mat.tsv",
    "co_occur_dist_data_file": "co_occurence_motif_dist_data.txt",
    "co_occur_mat_file": "co_occurence_mat.tsv",
    "co_occur_mat_norm_file": "co_occurence_mat.norm.tsv",
    "co_occur_network_fig": "co_occur_network.pdf",
    "motif_occurence_file": "final.motif_occurence.csv",
    "hamball_dir": "hamming_balls",
    "candidate_conseq_file": "candidate_conseq.csv",
    "final_conseq_file": "final_conseq.txt",
    "final_conseq_info_file": "final_conseq.info.csv",
    "sample_kmer_pkl_file": "sample_kmers.pkl",
    "sample_kmer_txt_file": "sample_kmers.tsv",
    "sample_kmer_hamdist_mat_file": "sample_kmer_hamdist_mat.pkl",
    "ld_data_file": "low_dim_data.tsv",
    "ld_fig_file_stem": "ld_data",
}


ti.set_logging_level(ti.ERROR)

MISSING_VAL = 255
GPU_MODE = False
ti.init(arch=ti.cpu, default_ip=ti.i64)
# ti.init(arch=ti.cuda, default_ip=ti.i64)
# if ti.cfg.arch == ti.cuda:
#     GPU_MODE = True
#     print("GPU is available")
# else:
#     GPU_MODE = False
#     print("GPU is not available")


@click.command(name="preproc")
@click.option(
    '--fasta_file',
    type=str,
    help='Input fasta file',
    required=True
    )
@click.option(
    '--res_dir',
    type=str,
    default=".",
    help='Result directory for storing all outputs',
    required=False
    )
@click.option(
    '--gpu_mode',
    type=bool,
    default=False,
    help='if GPU is available',
    required=False
    )
@click.option(
    '--debug',
    type=bool,
    default=False,
    help='display debug information.',
    required=False
    )
def preproc(fasta_file: str, res_dir=".", gpu_mode=False, debug=False):
    if gpu_mode:
        ti.init(arch=ti.cuda, default_ip=ti.i64)
    _preproc(fasta_file, res_dir, debug)


def read_default_config_file(debug=False):
    config_file_path = files(__package__).joinpath(FileNameDict["default_config_file"])  # may be in a zip file
    with as_file(config_file_path) as fh:
        with open(fh, "rb") as fh1:
            default_config_dict = tomllib.load(fh1)
    if debug:
        print(default_config_dict)
    return default_config_dict


def gen_motif_def_dict(config_dict: dict, debug=False) -> Dict:
    """
    generate the motif definition dict
    Args:
        config_dict: configuration dictionary derived from .toml file
        debug: if print debug information

    Returns:
        motif definition dictionary
    """
    motif_def_file = config_dict["motif_discovery"]["motif_def_file"]
    if motif_def_file == "default":
        motif_def_file_path = files(__package__).joinpath(FileNameDict["default_motif_def_file"])  # may be in a zip file
        with as_file(motif_def_file_path) as fh:
            motif_def_dict = init_motif_def_dict(fh, p_value_cutoff=config_dict["motif_discovery"]["p_value_cutoff"])
    else:
        assert Path(motif_def_file).exists()
        motif_def_dict = init_motif_def_dict(motif_def_file, p_value_cutoff=config_dict["motif_discovery"]["p_value_cutoff"])

    if debug:
        print(motif_def_dict)

    return motif_def_dict


def _preproc(fasta_file: str, res_dir=".", debug=False):
    input_fasta_file = fasta_file
    assert Path(input_fasta_file).exists()
    if not Path(res_dir).exists():
        Path(res_dir).mkdir()

    # read configuration file
    config_file_name = FileNameDict["config_file"]
    config_file_path = Path(res_dir) / config_file_name
    if Path(config_file_path).exists():
        with open(config_file_path, "rb") as fh:
            config_dict = tomllib.load(fh)
    else:
        config_dict = read_default_config_file(debug=debug)

    # save configuration file in output directory
    if not Path(config_file_path).exists() or config_dict["general"].get("input_fasta_file") is None:
        config_dict["general"]["input_fasta_file"] = input_fasta_file
        config_dict["general"]["res_dir"] = res_dir
        with open(config_file_path, 'wb') as fh:
            tomli_w.dump(config_dict, fh)

    # read motif definition file
    motif_def_dict = gen_motif_def_dict(config_dict, debug=debug)
    kmer_len_list = sorted([e for e in motif_def_dict if isinstance(e, int)])
    headers = motif_def_dict[kmer_len_list[0]].get_field_names()

    # save motif definition table in result directory
    motif_def_file = FileNameDict["motif_def_file"]
    motif_def_file_path = Path(res_dir) / motif_def_file
    with open(motif_def_file_path, "w+") as fh:
        fh.write(headers + "\n")
        for kmer_len in kmer_len_list:
            fh.write(str(motif_def_dict[kmer_len]) + "\n")

    # process input fasta file
    proc_input(config_dict["general"]["input_fasta_file"], config_dict["general"]["res_dir"],
               out_bin_file_name=FileNameDict["processed_fasta_file"],
               out_boarder_bin_file_name=FileNameDict["processed_fasta_seqboarder_file"],
               debug=debug)
    return config_dict, motif_def_dict


def proc_input(input_fasta_file: str, res_dir=".",
               out_bin_file_name: str="input.bin.pkl",
               out_boarder_bin_file_name: str="input.seqboarder.bin.pkl",
               debug=True):
    """
    process input fasta file, convert fasta file to a binary file "input.bin.pkl" in the output directory
    Args:
        input_fasta_file: input fasta file
        res_dir: output directory
        out_bin_file_name: name of the output binary file
        debug: if displaying debug information
    Returns: None
    """
    def get_file_size(fasta_file: str):
        n_seq, file_size = 0, 0
        for arr in read_dnaseq_file(fasta_file):
            n_seq += 1
            file_size = file_size + len(arr)
        return n_seq, file_size

    assert Path(input_fasta_file).exists()
    assert Path(res_dir).exists()
    assert out_bin_file_name.endswith(".pkl")

    n_seq, buffer_size = get_file_size(input_fasta_file)
    #input_binary_file = os.path.join(res_dir, out_bin_file_name)
    input_binary_file = str(Path(res_dir) / out_bin_file_name)
    input_boarder_file = str(Path(res_dir) / out_boarder_bin_file_name)

    if debug:
        print(f"Convert input file={input_fasta_file} into binary file {input_binary_file}. buffer_size={buffer_size/2**30}GB.")

    buffer = Buffer(buffer_size)
    # convert input fasta file into binary file
    convert_fasta_to_binary(input_fasta_file, buffer, n_seq, input_binary_file, input_boarder_file)

    print(f"input binary file {input_binary_file} generated.\n")


@dataclass
class MotifDef:
    kmer_len: int
    p_uniform: float
    max_ham_dist: int
    ratio_mu: float
    ratio_std: float
    ratio_cutoff: float

    @classmethod
    def get_field_names(cls):
        return ",".join(field.name for field in fields(cls))

    def __str__(self):
        return ",".join(str(getattr(self, field.name)) for field in fields(self))


def arr2dna(dna_np_arr: np.ndarray) -> str:
    base_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T', MISSING_VAL: 'N'}
    res = [base_map[i] for i in dna_np_arr]
    return "".join(res)


def dna2arr(dna_str, dtype=np.uint8, append_missing_val_flag=True) -> np.ndarray:
    """
    convert an input DNA string to numpy uint8 array, with a missing value appended
    Args:
        dna_str: a DNA sequence, all letters should be upper case
        dtype: data type for storing DNA string
        append_missing_val_flag: if a missing value is appended to the end of the array
    Returns:
        a numpy array
    """
    if append_missing_val_flag:
        res = np.empty(len(dna_str) + 1, dtype=dtype)
    else:
        res = np.empty(len(dna_str), dtype=dtype)
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i, b in enumerate(dna_str):
        res[i] = base_map.get(b, MISSING_VAL)
    if append_missing_val_flag:
        res[-1] = MISSING_VAL  # add a separator to the end of the string
    return res


def reverse_complement(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[base] for base in reversed(seq))


class Buffer:
    def __init__(self, buffer_size: int = 2 ** 26, dtype=np.uint8, id=None):
        # 2**26 bytes are 64MB, 2**30 bytes are 1GB
        self.buffer_size = buffer_size
        self.dtype = dtype
        self.buffer = np.full(buffer_size, MISSING_VAL, dtype=dtype)
        self.pointer = 0
        self.data_to_write = None
        self.is_full = False
        self.id = None

    def append(self, data: np.ndarray):
        # append input data into the buffer, return flag of success
        assert data.dtype == self.buffer.dtype
        data_size = len(data)
        buffer_space = self.buffer_size - self.pointer

        # Check if the buffer has enough space to accommodate the new data
        if data_size > buffer_space:
            self.data_to_write = data
            self.is_full = True
            return False
        else:
            # Append the data to the buffer
            self.buffer[self.pointer: (self.pointer + data_size)] = data
            self.pointer += data_size
            return True

    def flush(self):
        # flush the buffer such that it can be used again, data_to_write needs to be handled first
        assert self.data_to_write is None
        self.pointer = 0
        self.data_to_write = None
        self.is_full = False
        self.id = None


def read_dnaseq_file(file_name, file_type="fasta") -> np.ndarray:
    """
    file_name: input DNA sequence file name
    file_type: fasta, fastq,
    """

    def read_stream(fh):
        for rec in SeqIO.parse(fh, file_type):
            yield dna2arr(str(rec.seq).upper(), append_missing_val_flag=True)

    if file_name.endswith(".gz"):
        with gzip.open(file_name, "rt") as fh:
            yield from read_stream(fh)
    else:
        with open(file_name, "r") as fh:
            yield from read_stream(fh)


def convert_fasta_to_binary(fasta_file: str | Path, buffer: Buffer,
                            n_seq: int,
                            out_pkl_file: str | Path,
                            out_boarder_pkl_file: str | Path) -> None:
    def write_numpy_array(np_arr: np.ndarray, file_name: str | Path):
        assert str(file_name).endswith(".pkl")
        with open(file_name, "wb") as fh:
            pickle.dump(np_arr, fh)

    boarder_mat = np.zeros((n_seq, 2), dtype=int)

    pointer = 0
    for i, arr in enumerate(read_dnaseq_file(fasta_file)):
        flag = buffer.append(arr)
        # store the start and end position of each sequence
        boarder_mat[i][0] = pointer
        boarder_mat[i][1] = pointer + len(arr) - 1
        pointer += len(arr)
        assert flag
    assert not buffer.is_full
    write_numpy_array(buffer.buffer[0:buffer.pointer], out_pkl_file)
    write_numpy_array(boarder_mat, out_boarder_pkl_file)


# get the numpy dtype to store the kmer cnts
def get_cnt_dtype(kmer_len: int) -> np.dtype:
    if kmer_len < 16:
        return np.int32
    else:
        return np.int64


# get the hash dtype for given kmer length
def get_hash_dtype(kmer_len) -> np.dtype:
    if 0 < kmer_len < 16:
        return np.uint32
    elif kmer_len < 32:
        return np.uint64
    else:
        raise Exception(f"max_kmer_len=31, kmer_len={kmer_len} is greater the maximum value.")


# get the hash value for invalid kmers
def get_invalid_hash(dtype: Callable[[int], np.dtype]):
    return dtype(np.iinfo(dtype).max)


def my_parallel_sort(arr: np.ndarray):
    if arr.dtype == np.uint32:
        dtype = ti.u32
    elif arr.dtype == np.uint64:
        dtype = ti.u64
    else:
        raise Exception(f"unknown arr.dtype={arr.dtype}")

    tarr = ti.field(dtype=dtype, shape=len(arr))
    tarr.from_numpy(arr)
    parallel_sort(tarr)
    arr[:] = tarr.to_numpy()


def my_unique(hash_arr, hash_dtype, invalid_hash, cnt_dtype):
    my_parallel_sort(hash_arr)
    curr_hash = invalid_hash
    n_uniq_val = 0
    for i in range(len(hash_arr)):
        if hash_arr[i] == invalid_hash:
            break
        elif hash_arr[i] == curr_hash:
            continue
        else:
            curr_hash = hash_arr[i]
            n_uniq_val += 1
    unique_hash = np.empty(n_uniq_val, dtype=hash_dtype)
    counts = np.zeros(n_uniq_val, dtype=cnt_dtype)
    curr_hash = invalid_hash
    i_uniq = -1
    for i in range(len(hash_arr)):
        if hash_arr[i] == invalid_hash:
            break
        elif hash_arr[i] == curr_hash:
            counts[i_uniq] += 1
        else:
            curr_hash = hash_arr[i]
            i_uniq += 1
            unique_hash[i_uniq] = hash_arr[i]
            counts[i_uniq] += 1
    return unique_hash, counts


def kmer2hash(kmer: str) -> np.uint64:
    """
    kmer: input sequence
    return: a hash code
    """
    k = len(kmer)
    assert k < 32, "kmer should be shorted than 32 bases"
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    base = {bk: np.uint64(base_map[bk]) for bk in base_map}

    kh = base[kmer[0]]
    for tb in kmer[1:]:
        kh = kh << np.uint64(2)
        kh += base[tb]
    return kh


def hash2kmer(hashkey: np.uint64, k: int) -> str:
    """
    hashkey: hash key of kmer, numpy
    k: length of kmer
    """
    hashkey = np.uint64(hashkey)
    base = np.array('ACGT', 'c')
    arr = np.chararray(k)
    mask = np.uint64(3) # 0b11
    arr[-1] = base[ mask & hashkey]
    for i in range(2,k+1):
        hashkey = (hashkey >> np.uint64(2))
        arr[-i] = base[mask & hashkey]
    return arr.tobytes().decode("utf-8")


def comp_kmer_hash_taichi(seq_np_arr: np.ndarray, kmer_len: int) -> np.ndarray:
    """
    Compute kmer hash for each kmer from the input seq array
    Args:
        seq_np_arr: a numpy array that contain DNA sequences, A-0, C-1,
        kmer_len: length of kmer
    Returns: Tuple (uniq_kh_arr, uniq_kh_cnt_arr)
    """

    missing_val = np.uint8(MISSING_VAL)
    missing_val_arr = np.array([missing_val])
    seq_len = len(seq_np_arr)

    hash_dtype = get_hash_dtype(kmer_len)
    invalid_hash = get_invalid_hash(hash_dtype)
    invalid_hash_arr = np.array([invalid_hash])

    hash_arr = np.empty(seq_len, dtype=hash_dtype)
    if hash_dtype == np.uint32:
        kmer2hash_kernel_uint32(seq_np_arr, seq_len, kmer_len, hash_arr, invalid_hash_arr, missing_val_arr)
    elif hash_dtype == np.uint64:
        kmer2hash_kernel_uint64(seq_np_arr, seq_len, kmer_len, hash_arr, invalid_hash_arr, missing_val_arr)
    else:
        raise Exception(f"Unknown kmer hash type hash_dtype={hash_dtype}")
    return hash_arr


def count_uniq_hash(hash_arr: np.ndarray, kmer_len):
    hash_dtype = get_hash_dtype(kmer_len)
    invalid_hash = get_invalid_hash(hash_dtype)
    cnt_arr_dtype = get_cnt_dtype(kmer_len)

    if GPU_MODE:
        unique_hash, counts = my_unique(hash_arr, hash_dtype, invalid_hash, cnt_arr_dtype)
    else:
        unique_hash, counts = np.unique(hash_arr, return_counts=True)
        inds = unique_hash != invalid_hash
        unique_hash = unique_hash[inds]
        counts = counts[inds].astype(cnt_arr_dtype)
    # hash_counts_dict = Counter(dict(zip(unique_hash, counts)))

    # return hash_counts_dict
    return unique_hash, counts


def cal_hamming_dist(kh_arr: np.ndarray, consensus_kh: np.uint64, kmer_len: int) -> np.ndarray:
    """
    calculate the Hamming distances between each element in kh_arr and the consensus sequence
    Args:
        kh_arr: kmer hash array
        consensus_kh: kmer hash of the consensus sequence
        kmer_len: kmer length
    Returns:
        Hamming distance array, np.ndarray object
    """
    ham_dist_arr = np.empty_like(kh_arr, dtype=np.uint8)
    hash_dtype = get_hash_dtype(kmer_len)
    consensus_kh_arr = np.array([consensus_kh], dtype=hash_dtype)
    hash_arr_size = len(kh_arr)

    if hash_dtype == np.uint32:
        cal_ham_dist_kernel_uint32(kh_arr, consensus_kh_arr, ham_dist_arr, hash_arr_size, kmer_len)
    elif hash_dtype == np.uint64:
        cal_ham_dist_kernel_uint64(kh_arr, consensus_kh_arr, ham_dist_arr, hash_arr_size, kmer_len)
    else:
        raise Exception(f"Unknown kmer hash type hash_dtype={hash_dtype}")
    return ham_dist_arr


def cal_hamming_dist_head(kh_arr: np.ndarray, consensus_kh: np.uint64, kmer_len: int, consensus_len: int) -> np.ndarray:
    """
    calculate the Hamming distances between each element in kh_arr and the consensus sequence
    The consensus sequence might be shorter than the kmer length
    Only match the consensus from the head of each kmer
    Args:
        kh_arr: kmer hash array
        consensus_kh: kmer hash of the consensus sequence
        kmer_len: kmer length
        consensus_len: length of the consensus sequence
    Returns:
        Hamming distance array, np.ndarray object
    """
    assert consensus_len <= kmer_len

    ham_dist_arr = np.empty_like(kh_arr, dtype=np.uint8)
    hash_dtype = get_hash_dtype(kmer_len)
    consensus_kh_arr = np.array([consensus_kh], dtype=hash_dtype)
    hash_arr_size = len(kh_arr)

    if hash_dtype == np.uint32:
        cal_partial_ham_dist_head_kernel_uint32(kh_arr, consensus_kh_arr, ham_dist_arr, hash_arr_size, kmer_len,
                                                consensus_len)
    elif hash_dtype == np.uint64:
        cal_partial_ham_dist_head_kernel_uint64(kh_arr, consensus_kh_arr, ham_dist_arr, hash_arr_size, kmer_len,
                                                consensus_len)
    else:
        raise Exception(f"Unknown kmer hash type hash_dtype={hash_dtype}")
    return ham_dist_arr


def cal_hamming_dist_tail(kh_arr: np.ndarray, consensus_kh: np.uint64, kmer_len: int, consensus_len: int) -> np.ndarray:
    """
    calculate the Hamming distances between each element in kh_arr and the consensus sequence
    The consensus sequence might be shorter than the kmer length
    Only match the consensus from the tail of each kmer
    Args:
        kh_arr: kmer hash array
        consensus_kh: kmer hash of the consensus sequence
        kmer_len: kmer length
        consensus_len: length of the consensus sequence
    Returns:
        Hamming distance array, np.ndarray object
    """
    assert consensus_len <= kmer_len

    ham_dist_arr = np.empty_like(kh_arr, dtype=np.uint8)
    hash_dtype = get_hash_dtype(kmer_len)
    consensus_kh_arr = np.array([consensus_kh], dtype=hash_dtype)
    hash_arr_size = len(kh_arr)

    if hash_dtype == np.uint32:
        cal_partial_ham_dist_tail_kernel_uint32(kh_arr, consensus_kh_arr, ham_dist_arr, hash_arr_size, kmer_len,
                                                consensus_len)
    elif hash_dtype == np.uint64:
        cal_partial_ham_dist_tail_kernel_uint64(kh_arr, consensus_kh_arr, ham_dist_arr, hash_arr_size, kmer_len,
                                                consensus_len)
    else:
        raise Exception(f"Unknown kmer hash type hash_dtype={hash_dtype}")
    return ham_dist_arr


def mask_input(seq_np_arr: np.ndarray, kmer_len: int, consensus_kh_arr: np.ndarray, max_hamball_dist_arr: np.ndarray):
    """
    mask the input seq_np_arr such that occurences of all kmers in given consensus hamming ball as masked
    consensus_kh_arr can contain revcom, repetitive kmers
    Args:
        seq_np_arr: converted numpy arr of the input DNA sequence
        kmer_len: kmer length
        consensus_kh_arr: consensus hash arr, all consensus must be the same length
        max_hamball_dist_arr: maximum Hamming distance in the Hamming ball for the corresponding consensus hash
    Returns:
        seq_np_arr with relevant positions in seq_np_arr overwritten as Missing value
    """
    def _mask_one_hamball(seq_np_arr, kh_hash_arr, consensus_kh, max_hamball_dist, kmer_len):
        ham_dist_arr = cal_hamming_dist(kh_hash_arr, consensus_kh, kmer_len)
        flag_arr = ham_dist_arr <= max_hamball_dist
        for i, flag in enumerate(flag_arr):
            if flag:
                j = i + kmer_len if i + kmer_len < len(seq_np_arr) else len(seq_np_arr)
                seq_np_arr[i:j] = MISSING_VAL
        del ham_dist_arr

    kh_hash_arr = comp_kmer_hash_taichi(seq_np_arr, kmer_len)
    for consensus_kh, max_hamball_dist in zip(consensus_kh_arr, max_hamball_dist_arr):
        _mask_one_hamball(seq_np_arr, kh_hash_arr, consensus_kh, max_hamball_dist, kmer_len)

    del kh_hash_arr
    return seq_np_arr


def get_revcom_hash_arr(in_hash_arr: np.ndarray, kmer_len: int) -> np.ndarray:
    hash_dtype = get_hash_dtype(kmer_len)
    mask_arr = np.array([(1 << 2 * kmer_len) - 1, 3], dtype=hash_dtype) # mask and twobit_mask

    out_hash_arr = np.empty_like(in_hash_arr)
    hash_arr_size = len(in_hash_arr)
    if hash_dtype == np.uint32:
        revcom_hash_kernel_uint32(in_hash_arr, out_hash_arr, mask_arr, kmer_len, hash_arr_size)
    elif hash_dtype == np.uint64:
        revcom_hash_kernel_uint64(in_hash_arr, out_hash_arr, mask_arr, kmer_len, hash_arr_size)
    return out_hash_arr


def revcom_hash(in_hash: np.uint64, kmer_len: int) -> np.uint64 | np.uint32:
    hash_dtype = get_hash_dtype(kmer_len)
    in_hash = hash_dtype(in_hash)
    mask_arr = np.array([(1 << 2 * kmer_len) - 1, 3], dtype=hash_dtype) # mask and twobit_mask
    mask = mask_arr[0]
    twobit_mask = mask_arr[1]

    com_hash = mask - in_hash  # complement hash
    # reverse
    ret_hash = twobit_mask & com_hash
    for i in range(kmer_len - 1):
        ret_hash = ret_hash << hash_dtype(2)
        com_hash = com_hash >> hash_dtype(2)
        ret_hash += twobit_mask & com_hash
    return ret_hash


def merge_revcom(uniq_kmer_hash_arr: np.ndarray, uniq_kh_cnt_arr: np.ndarray,
                 kmer_len: int, keep_lower_hash_flag=True) -> Tuple:
    """
    merge reverse complements by summing the counts of a pair.
    Only keep the smaller kmer hash for a pair of revcoms. Palindrome's counts keep the same.
    Args:
        uniq_kmer_hash_arr: unique kmer hash array sorted in ascending order
        uniq_kh_cnt_arr: counts of the corresponding kmer hash in uniq_kmer_hash_arr
        kmer_len: kmer length
        keep_lower_hash_flag: if keeping the lower hash as the key when merging a pair of reverse complements
    Returns:
        a tuple (kh_arr, kh_cnt_arr) with revcom merged
    """
    revcom_uniq_kmer_hash_arr = get_revcom_hash_arr(uniq_kmer_hash_arr, kmer_len)

    # merge kh counts with its revcom, both comm_kh_nat_inds and comm_kh_rc_inds should be the same set, but in different order
    comm_kh, comm_kh_nat_inds, comm_kh_rc_inds = np.intersect1d(uniq_kmer_hash_arr, revcom_uniq_kmer_hash_arr,
                                                                return_indices=True)
    uniq_kh_cnt_arr[comm_kh_nat_inds] += uniq_kh_cnt_arr[comm_kh_rc_inds]

    # handle palindromes, only count once for palindrome
    palindrome_inds = np.where(uniq_kmer_hash_arr == revcom_uniq_kmer_hash_arr)[0]
    uniq_kh_cnt_arr[palindrome_inds] = uniq_kh_cnt_arr[palindrome_inds] / 2

    # remove kmers that are revcom
    if keep_lower_hash_flag:
        inds = uniq_kmer_hash_arr[comm_kh_nat_inds] > revcom_uniq_kmer_hash_arr[comm_kh_nat_inds]
    else:
        inds = uniq_kmer_hash_arr[comm_kh_nat_inds] < revcom_uniq_kmer_hash_arr[comm_kh_nat_inds]

    other_inds = comm_kh_nat_inds[inds]
    uniq_kmer_hash_arr = np.delete(uniq_kmer_hash_arr, other_inds)
    revcom_uniq_kmer_hash_arr = np.delete(revcom_uniq_kmer_hash_arr, other_inds)
    uniq_kh_cnt_arr = np.delete(uniq_kh_cnt_arr, other_inds)

    # replace kmers that has a high/low hash in the list with low/high hash depending on keep_lower_hash_flag
    if keep_lower_hash_flag:
        inds = uniq_kmer_hash_arr > revcom_uniq_kmer_hash_arr
    else:
        inds = uniq_kmer_hash_arr < revcom_uniq_kmer_hash_arr
    uniq_kmer_hash_arr[inds] = revcom_uniq_kmer_hash_arr[inds]

    return uniq_kmer_hash_arr, uniq_kh_cnt_arr


def mask_ham_ball(seq_np_arr: np.ndarray, motif_def_dict: dict,
                    consensus_seq_list: List[str], max_ham_dist_list: List[int] = ()) -> np.ndarray:
    """
    mosk consensus's hamming ball given by the user
    Args:
        seq_np_arr: numpy array of the input sequence
        motif_def_dict: motif definition dictionary, kmer_len : MotifDef obj
        consensus_seq_list: consensus sequence list provided by the user
        max_ham_dist_list: maximum hamming distance for each consensus sequence
    Returns:
        masked sequence array
    """

    len_list = np.array([len(conseq) for conseq in consensus_seq_list])
    if len(max_ham_dist_list) == 0:
        # user only provide consensus sequence, but not max_ham_dist, we derive from motif_def_table
        max_ham_dist_list = [motif_def_dict[conseq_len].max_ham_dist for conseq_len in len_list]
    assert len(max_ham_dist_list) == len(consensus_seq_list)

    uniq_len_arr = np.unique(len_list)
    conseq_kh_arr_by_len = []
    conseq_max_hamdist_by_len = []
    for uniq_len in uniq_len_arr:
        inds = np.where(len_list == uniq_len)[0]
        tmp_list = [consensus_seq_list[i] for i in inds]
        tmp_kh_list = np.array([kmer2hash(seq) for seq in tmp_list])
        tmp_len_arr = np.array([max_ham_dist_list[i] for i in inds])
        conseq_kh_arr_by_len.append(tmp_kh_list)
        conseq_max_hamdist_by_len.append(tmp_len_arr)

    for uniq_len, conseq_kh_arr, hamdist_arr \
            in zip(uniq_len_arr, conseq_kh_arr_by_len, conseq_max_hamdist_by_len):
        kmer_len = uniq_len
        seq_np_arr = mask_input(seq_np_arr, kmer_len, conseq_kh_arr, hamdist_arr)

    return seq_np_arr


def init_motif_def_dict(motif_def_file, p_value_cutoff=1e-10) -> dict:
    motif_def_dict = {}
    motif_def_dict["p_value_cutoff"] = p_value_cutoff

    df = pd.read_csv(motif_def_file)
    for ind, row in df.iterrows():
        kmer_len = int(row['kmer_len'])
        p_uniform = row['p_uniform']
        max_ham_dist = int(row['max_ham_dist'])
        ratio_mu = row["ratio_mu"]
        ratio_std = row["ratio_std"]
        ratio_cutoff = norm.ppf(1-p_value_cutoff, loc=ratio_mu, scale=ratio_std)
        motif_def_dict[kmer_len] = MotifDef(kmer_len, p_uniform, max_ham_dist, ratio_mu, ratio_std, ratio_cutoff)

    return motif_def_dict


def remove_duplicate_hash_per_seq(hash_arr: np.array, boarder_mat: np.ndarray, invalid_hash: int) -> np.array:
    """
    remove duplicate hash from the hash array of each sequence
    Args:
        hash_arr: 1d array, the combined hash array of all sequences, seq separated by invalid hash
        boarder_mat: n x 2 array, each row is a sequence, first column is start index, second column is end index
        invalid_hash: invalid hash

    Returns:
        None
    """
    assert boarder_mat.shape[1] == 2
    for st, en in boarder_mat:
        tmparr = np.full(en-st, invalid_hash, dtype=hash_arr.dtype)
        tmp_uniq_vals, inds = np.unique(hash_arr[st:en], return_index=True)
        tmparr[inds] = tmp_uniq_vals
        hash_arr[st:en] = tmparr[:]
    return hash_arr





if __name__ == "__main__":
    #test_process_input()
    #test_comp_kmer_hash_taichi()
    #test_count_uniq_hash()
    #test_mask_input()
    #test_merge_revcom()
    #test_hash2kmer()
    #test_kmer2hash()

    #test_find_motif()
    #test_init_motif_def_dict()
    #test_mask_ham_ball()
    pass

    # import argparse
    #
    # parser = argparse.ArgumentParser(description='Process fasta files.')
    # parser.add_argument('--input_fasta_file', type=str, help='Path to the directory containing fasta files')
    # parser.add_argument("--min_k", type=int, help="Minimum kmer length.")
    # parser.add_argument("--max_k", type=int, help="Maximum kmer length.")
    # parser.add_argument("--preproc", action="store_true", help="Enable preprocessing.")
    # parser.add_argument("--net_flag", action="store_false", help="Enable preprocessing.")
    # parser.add_argument("-f", "--user_motif_file", default="", help="Path to the .txt file containing DNA sequences.")
    # args = parser.parse_args()
    # u_motif = read_user_motif_file(args.user_motif_file)
    #
    #
    # fasta_file = "../../data/raw/test.fasta"
    # min_k = 5
    # max_k = 10
    # u = ['AAAAA', 'AAAAAA', 'AAAAAAA']
    # preproc = True
    # main(args.path, u_motif, args.min_k, args.max_k, args.preproc)
    # #main(fasta_file, u, min_k, max_k, preproc)

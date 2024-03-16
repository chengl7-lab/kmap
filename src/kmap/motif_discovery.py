import matplotlib.pyplot as plt

from .kmer_count import (comp_kmer_hash_taichi, count_uniq_hash, merge_revcom,
                                 cal_hamming_dist, revcom_hash, mask_input, proc_input,
                                 init_motif_def_dict, mask_ham_ball, hash2kmer, kmer2hash,
                                 cal_hamming_dist_head, cal_hamming_dist_tail, dna2arr,
                                 reverse_complement, get_hash_dtype, FileNameDict, gen_motif_def_dict)
import numpy as np
import pickle
from scipy.stats import norm
from typing import List, Tuple
import warnings

from pathlib import Path
from Bio import SeqIO
import gzip
import tomllib
import logomaker
import pandas as pd
import click


@click.command(name="scan_motif")
@click.option(
    '--res_dir',
    type=str,
    help='Result directory for storing all outputs',
    required=True
    )
@click.option(
    '--debug',
    type=bool,
    default=False,
    help='display debug information.',
    required=False
    )
def scan_motif(res_dir: str, debug=False):
    _scan_motif(res_dir, debug)


@click.command(name="draw_logo")
@click.option(
    '--cnt_mat_numpy_file',
    type=str,
    help='count matrix file',
    required=True
    )
@click.option(
    '--output_fig_file',
    type=str,
    default="motif_logo.png",
    help='output figure file name.',
    required=False
    )
def draw_logo(cnt_mat_numpy_file: str, output_fig_file=None):
    _draw_logo(cnt_mat_numpy_file, output_fig_file)


@click.command(name="ex_hamball")
@click.option(
    '--res_dir',
    type=str,
    help='Result directory for storing all outputs',
    required=True
    )
@click.option(
    '--conseq',
    type=str,
    help='the consensus sequence',
    required=True
    )
@click.option(
    '--return_type',
    type=str,
    help='output file form, can be ["hash" | "kmer" | "matrix"]',
    required=True
    )
@click.option(
    '--output_file',
    type=str,
    help='output file name, including the suffix',
    required=True
    )
@click.option(
    '--max_ham_dist',
    type=int,
    default=-1,
    help='The radius of the Hamming ball. -1 means taking the radius from motif_def_table.csv',
    required=False
    )
def ex_hamball(res_dir: str, conseq: str, return_type: str, output_file: str,
               max_ham_dist: int=-1):
    _ex_hamball(res_dir, conseq, return_type, output_file, max_ham_dist)


def write_lines(str_list: List, outfile: str|Path):
    with open(outfile, 'w+') as fh:
        for line in str_list:
            fh.write(line + "\n")


def _scan_motif(res_dir: str, debug=False):
    config_file_name = FileNameDict["config_file"] # config.toml
    config_file_path = Path(res_dir) / config_file_name

    motif_def_file = FileNameDict["motif_def_file"] # motif_def_table.csv
    motif_def_file_path = Path(res_dir) / motif_def_file

    proc_fasta_file = FileNameDict["processed_fasta_file"] # input.bin.pkl
    proc_fasta_file_path = Path(res_dir) / proc_fasta_file

    assert config_file_path.exists()
    assert motif_def_file_path.exists()
    assert proc_fasta_file_path.exists()

    # load config and motif_def files
    with open(config_file_path, "rb") as fh:
        config_dict = tomllib.load(fh)
    motif_def_dict = gen_motif_def_dict(config_dict, debug=debug)
    min_k = config_dict["kmer_count"]["min_k"]
    max_k = config_dict["kmer_count"]["max_k"]
    revcom_mode = config_dict["kmer_count"]["revcom_mode"]

    mask_noise_seq_list = []
    if config_dict["motif_discovery"]["noise_kmer_file"] != "None":
        noise_kmer_file = config_dict["motif_discovery"]["noise_kmer_file"]
        assert Path(noise_kmer_file).exists()
        with open(Path(noise_kmer_file), "r") as fh:
            for line in fh:
                line = line.strip()
                if len(line) > 0:
                    mask_noise_seq_list.append(line)

    with open(proc_fasta_file_path, "rb") as fh:
        seq_np_arr = pickle.load(fh)

    # mask user provided noise sequences from the input sequence
    if len(mask_noise_seq_list) > 0:
        max_ham_dist_list = [0 for _ in mask_noise_seq_list]
        seq_np_arr = mask_ham_ball(seq_np_arr, motif_def_dict, mask_noise_seq_list, max_ham_dist_list)

    # load necessary parameters
    top_k = config_dict["motif_discovery"]["top_k"]
    n_trial = config_dict["motif_discovery"]["n_trial"]
    save_kmer_cnt_flag = config_dict["motif_discovery"]["save_kmer_cnt_flag"]
    orig_seq_np_arr = seq_np_arr.copy()
    candidate_conseq_list = []

    if save_kmer_cnt_flag:
        kmer_count_dir = Path(res_dir) / FileNameDict["kmer_count_dir"]
        if not kmer_count_dir.exists():
            kmer_count_dir.mkdir()

    # candidate motif sequences
    candidate_conseq_file = Path(res_dir) / FileNameDict["candidate_conseq_file"]
    if candidate_conseq_file.exists():
        print(f"{candidate_conseq_file} already exist, re-use it.")
    else:
        # motif discovery
        res = ["kmer_len,conseq_hash,conseq,hamball_proportion,hamball_ratio,log10_p_value"]
        for kmer_len in range(min_k, max_k + 1):
            seq_np_arr[:] = orig_seq_np_arr[:]
            p_uniform_k = motif_def_dict[kmer_len].p_uniform
            max_ham_dist = motif_def_dict[kmer_len].max_ham_dist
            ratio_mu = motif_def_dict[kmer_len].ratio_mu
            ratio_std = motif_def_dict[kmer_len].ratio_std
            ratio_cutoff = motif_def_dict[kmer_len].ratio_cutoff

            kmer_cnt_file = Path(res_dir) / FileNameDict[
                "kmer_count_dir"] / f"k{kmer_len}.pkl"  # [kmer_len, uniq_kh_arr, uniq_kh_cnt_arr]
            consensus_kh_dict = find_motif(seq_np_arr, kmer_len, max_ham_dist, p_uniform_k,
                                           ratio_mu, ratio_std, ratio_cutoff,
                                           top_k, n_trial,
                                           revcom_mode, save_kmer_cnt_flag=save_kmer_cnt_flag,
                                           kmer_cnt_pkl_file=kmer_cnt_file,
                                           debug=debug)  # snp_np_arr are mutated in find_motif()
            if debug:
                print(f"filtered consensus kmers when k = {kmer_len}")
            for kh, values in consensus_kh_dict.items():
                prop, ratio, log10_p_value = values
                kmer_seq = hash2kmer(kh, kmer_len)
                if debug:
                    print(
                        f'{hash2kmer(kh, kmer_len)} perc={prop * 100:0.5f}%  hamball_ratio={ratio} log10_p_value={log10_p_value}')
                res.append(f"{kmer_len},{kh},{kmer_seq},{prop:0.8f},{ratio:0.4f},{log10_p_value:0.4f}")
                candidate_conseq_list.append(kmer_seq)
        print(f"kmer counting finished for k={min_k}...{max_k}. Candidate consensus sequences generated.")
        write_lines(res, candidate_conseq_file)

    # merge candidate motif sequences
    final_conseq_file = Path(res_dir) / FileNameDict["final_conseq_file"]
    if final_conseq_file.exists():
        with open(final_conseq_file, "r") as fh:
            final_conseq_list = fh.read().splitlines()
        print(f"{final_conseq_file} already exist, re-use it.")
    else:
        final_conseq_list = merge_consensus_seqs(candidate_conseq_list)
        write_lines(final_conseq_list, final_conseq_file)

    final_conseq_info_file = Path(res_dir) / FileNameDict["final_conseq_info_file"]
    if final_conseq_info_file.exists():
        print(f"{final_conseq_info_file} already exist, re-use it.")
    else:
        with open(final_conseq_file, "r") as fh:
            final_conseq_list = fh.read().splitlines()
        with open(candidate_conseq_file, "r") as fh:
            candidate_conseq_info_list = fh.read().splitlines()
        final_conseq_info_list = [candidate_conseq_info_list[0]] # header
        for conseq in final_conseq_list:
            for line in candidate_conseq_info_list:
                if ","+conseq+"," in line:
                    final_conseq_info_list.append(line)
                    continue
        write_lines(final_conseq_info_list, final_conseq_info_file)
    print("Final consensus sequences generated.")

    # generate motif co-occurence matrix
    co_occur_mat_file = Path(res_dir) / FileNameDict["co_occur_mat_file"]
    co_occur_mat_norm_file = Path(res_dir) / FileNameDict["co_occur_mat_norm_file"]
    if co_occur_mat_file.exists():
        print(f"{co_occur_mat_file}, re-use it!")
    else:
        min_motif_cnt = config_dict["motif_discovery"]["min_motif_cnt"]
        input_fasta_file = config_dict["general"]["input_fasta_file"]
        co_occur_mat = gen_motif_co_occurence_mat(input_fasta_file, final_conseq_list,
                                                  motif_def_dict, min_motif_cnt=min_motif_cnt, revcom_mode=revcom_mode)
        co_occur_norm_mat = co_occur_mat / np.diag(co_occur_mat).reshape((-1, 1))
        with open(co_occur_mat_file, "w+") as fh:
            np.savetxt(fh, co_occur_mat, delimiter="\t", fmt="%d")
        with open(co_occur_mat_norm_file, "w+") as fh:
            np.savetxt(fh, co_occur_norm_mat, delimiter="\t", fmt="%2.3f")
    print("motif co-occurence matrix generated.")

    # sample kmers
    if not save_kmer_cnt_flag:
        print(f"kmers cannot be sampled when {save_kmer_cnt_flag=}, quit!")
        return
    n_total_sample = config_dict["motif_discovery"]["n_total_sample"]
    n_motif_sample = config_dict["motif_discovery"]["n_motif_sample"]
    kmer_count_dir = Path(res_dir) / FileNameDict["kmer_count_dir"]
    kmer_len = max([len(conseq) for conseq in final_conseq_list])
    (samp_kh_arr, samp_cnts, samp_label_arr, conseq_list) \
        = sample_disp_kmer(final_conseq_list, kmer_len, motif_def_dict,
            kmer_count_dir=kmer_count_dir, n_total_sample = n_total_sample,
            n_motif_kmer = n_motif_sample, revcom_mode = revcom_mode)
    sample_kmer_pkl_file = Path(res_dir) / FileNameDict["sample_kmer_pkl_file"]
    sample_kmer_txt_file = Path(res_dir) / FileNameDict["sample_kmer_txt_file"]
    with open(sample_kmer_pkl_file, "wb") as fh:
        pickle.dump([samp_kh_arr, samp_cnts, samp_label_arr, conseq_list], fh)
    sample_kmer_lines = []
    for kh, cnt, label in zip(samp_kh_arr, samp_cnts, samp_label_arr):
        for _ in range(cnt):
            sample_kmer_lines.append(f"{hash2kmer(kh, kmer_len)}\t{label}")
    write_lines(sample_kmer_lines, sample_kmer_txt_file)
    print(f"kmers are sampled for visualization. {kmer_len= }, {n_total_sample= }, {n_motif_sample= }")

    # calculate Hamming distance between sampled kmers
    hamdist_mat = cal_samp_kmer_hamdist_mat(samp_kh_arr, samp_cnts, samp_label_arr, conseq_list, kmer_len,
                                            uniq_dist_flag = False)
    label_arr = _convert_to_block_arr(samp_label_arr, samp_cnts)
    sample_kmer_hamdist_mat_file = Path(res_dir) / FileNameDict["sample_kmer_hamdist_mat_file"]
    with open(sample_kmer_hamdist_mat_file, "wb") as fh:
        pickle.dump([kmer_len, hamdist_mat, label_arr], fh)
    print(f"Hamming distance matrix of sampled kmers are generated.")

    gen_hamball_flag = config_dict["motif_discovery"]["gen_hamball_flag"]
    if gen_hamball_flag:
        for i, conseq in enumerate(final_conseq_list):
            if debug:
                print(f"generating motif count matrix and draw logo for motif {i}: {conseq}")
            out_dir_path = Path(res_dir) / FileNameDict["hamball_dir"]
            if not out_dir_path.exists():
                out_dir_path.mkdir()
            return_type = "matrix"
            output_cntmat_file = str(out_dir_path / f"cntmat_motif{i}_{conseq}.csv")
            max_ham_dist = motif_def_dict[len(conseq)].max_ham_dist
            _ex_hamball(res_dir, conseq, return_type, output_cntmat_file, max_ham_dist=max_ham_dist)
            output_logo_file = str(out_dir_path / f"logo_motif{i}_{conseq}.pdf")
            _draw_logo(output_cntmat_file, output_fig_file = output_logo_file)
        print("Motif count matrix and logo extracted.")

    print(f"All tasks of scan motif finished.")


def _ex_hamball(res_dir: str, conseq: str, return_type: str, output_file: str,
               max_ham_dist: int=-1):
    """
    Extract kmers of a Hamming ball
    Args:
        res_dir: result directory
        conseq: consensus sequence
        return_type: "hash" | "kmer" | "matrix"
        output_file: output file name
        max_ham_dist: maximum Hamming distance of the Hamming ball
        revcom_mode: reverse complement mode
    Returns:
        Tuple (hash,cnt) | (kmer, cnt) | matrix
    """
    config_file_name = FileNameDict["config_file"]  # config.toml
    config_file_path = Path(res_dir) / config_file_name
    assert config_file_path.exists()

    # load config and motif_def files
    with open(config_file_path, "rb") as fh:
        config_dict = tomllib.load(fh)

    assert return_type in ("hash", "kmer", "matrix")
    motif_def_file = FileNameDict["motif_def_file"]  # motif_def_table.csv
    motif_def_file_path = Path(res_dir) / motif_def_file
    revcom_mode = config_dict["kmer_count"]["revcom_mode"]

    uniq_kh_arr, uniq_kh_cnt_arr = ex_hamball_kh_arr(res_dir, conseq, max_ham_dist, motif_def_file_path, revcom_mode)
    kmer_len = len(conseq)

    with open(output_file, 'w+') as fh:
        if return_type == "hash":
            for kh, cnt in zip(uniq_kh_arr, uniq_kh_cnt_arr):
                fh.write(f"{kh},{cnt}\n")
        elif return_type == "kmer":
            for kh, cnt in zip(uniq_kh_arr, uniq_kh_cnt_arr):
                fh.write(f"{hash2kmer(kh, kmer_len)},{cnt}\n")
        else:
            cnt_mat = cal_cnt_mat(uniq_kh_arr, uniq_kh_cnt_arr, kmer_len)
            np.savetxt(fh, cnt_mat, delimiter=",", fmt="%d")

    print(f"Extract Hamming ball [type={return_type}] save in {output_file}.")


def merge_consensus_seqs(conseq_list: List[str]) -> List[str]:
    """
    merge motifs of different lengths
    Args:
        conseq_list: all candidate motif consensus sequences of different lengths
    Returns:
        final consensus sequences
    """

    def _overlap(long_kmer, short_kmer):
        # if short_kmer is a substring of long_kmer
        len_l, len_s = len(long_kmer), len(short_kmer)
        assert len_l >= len_s
        for i in range(len_l - len_s + 1):
            if short_kmer == long_kmer[i:i+len_s]:
                return True
        return False

    def _overlap_shift_one(long_kmer, short_kmer):
        # if k-1 substring of short_kmer is also a subtring of long_kmer
        return _overlap(long_kmer, short_kmer[:-1]) or _overlap(long_kmer, short_kmer[1:])

    # Sort the k-mers in descending order of their length
    conseq_list = sorted(conseq_list, key=len, reverse=True)
    final_conseq_list = []

    while len(conseq_list) > 0:
        curr_conseq = conseq_list[0]
        rc_curr_conseq = reverse_complement(curr_conseq)

        conseq_len_list = [len(conseq) for conseq in conseq_list]
        sub_inds_1 = [i for i, seq_len in enumerate(conseq_len_list) if seq_len == (len(curr_conseq) - 1)]
        sub_inds_2 = [i for i, seq_len in enumerate(conseq_len_list) if seq_len == (len(curr_conseq) - 2)]

        substr1 = None
        for i1 in sub_inds_1:
            if _overlap_shift_one(curr_conseq, conseq_list[i1]) or _overlap_shift_one(rc_curr_conseq, conseq_list[i1]):
                substr1 = conseq_list[i1]
                break
        substr2 = None
        for i2 in sub_inds_2:
            if _overlap_shift_one(curr_conseq, conseq_list[i2]) or _overlap_shift_one(rc_curr_conseq, conseq_list[i2]):
                substr2 = conseq_list[i2]
                break

        if substr1 and substr2:
            final_conseq_list.append(substr1)
            # remove all substrings
            new_conseq_list = []
            for i, conseq in enumerate(conseq_list):
                if _overlap_shift_one(curr_conseq, conseq) or _overlap_shift_one(rc_curr_conseq, conseq):
                    continue
                else:
                    new_conseq_list.append(conseq)
            conseq_list = new_conseq_list
        else:
            conseq_list = conseq_list[1:]

    return final_conseq_list


def find_motif(seq_np_arr, kmer_len: int, max_ham_dist, p_unif,
               ratio_mu, ratio_std, ratio_cutoff, top_k=5, n_trial=10,
               merge_revcom_mode=True, save_kmer_cnt_flag=True,
               kmer_cnt_pkl_file: Path=None, debug=False) -> dict:
    """
    main motif discovery code,
        step 0: we try to pick the largest hamming ball for top k kmers each time,
        step 1: check if it passes the significant test
        step 3: repeat the process n_trial times
    Args:
        seq_np_arr: input sequence numpy array (uint8), missing values are 255
        kmer_len: kmer length
        max_ham_dist: maximum hamming ball distance for the given kmer length
        p_uniform: probability of a hamming ball centered on a random kmer
        ratio_mu: mean of the Hamming ball ratio distribution
        ratio_std: std of the Hamming ball ratio distribution
        ratio_cutoff: Hamming ball ratio cutoff for a kmer to be considered as significant
        top_k: top k consensus sequences to consider
        n_trial: number of times we try to pick up a motif
        merge_revcom_mode: if revcom should be merged
    Returns:
        dict, key is consensus_kmer_hash, value is a tuple of the Hamming ball (proportion, ratio, log10_pvalue)
    """

    if save_kmer_cnt_flag and kmer_cnt_pkl_file and Path(kmer_cnt_pkl_file).exists():
        with open(Path(kmer_cnt_pkl_file), "rb") as fh:
            kmer_len_from_pkl_file, uniq_kh_arr, uniq_kh_cnt_arr = pickle.load(fh)
            assert kmer_len == kmer_len_from_pkl_file
    else:
        # first round
        hash_arr = comp_kmer_hash_taichi(seq_np_arr, kmer_len)
        uniq_kh_arr, uniq_kh_cnt_arr = count_uniq_hash(hash_arr, kmer_len)
        # merge revcom
        if merge_revcom_mode:
            uniq_kh_arr, uniq_kh_cnt_arr = merge_revcom(uniq_kh_arr, uniq_kh_cnt_arr, kmer_len,
                                                        keep_lower_hash_flag=True)

    if save_kmer_cnt_flag and kmer_cnt_pkl_file and not Path(kmer_cnt_pkl_file).exists():
        assert kmer_cnt_pkl_file
        with open(kmer_cnt_pkl_file, "wb") as fh:
           pickle.dump([kmer_len, uniq_kh_arr, uniq_kh_cnt_arr], fh)

    # count total kmer excluding invalid kmer
    n_total_kmer = sum(uniq_kh_cnt_arr)

    res_consensus_kh_list = []
    res_consensus_proportion_list = []
    res_consensus_ratio_list = []
    res_consensus_log10_pvalue_list =[]

    for i_trial in range(n_trial):
        # get the kmer with maximum hamming ball counts
        top_k_inds = np.array(np.argpartition(uniq_kh_cnt_arr, -top_k)[-top_k:])
        if len(top_k_inds) == 0:
            break

        hamball_cnt_arr = np.zeros(top_k)
        for i, ind in enumerate(top_k_inds):
            kh = uniq_kh_arr[ind]
            dist_arr = cal_hamming_dist(uniq_kh_arr, kh, kmer_len)
            if merge_revcom_mode:
                rc_kh = revcom_hash(kh, kmer_len)
                rc_dist_arr = cal_hamming_dist(uniq_kh_arr, rc_kh, kmer_len)  # revcom
                dist_arr = np.minimum(dist_arr, rc_dist_arr)
            hamball_cnt_arr[i] = np.sum(uniq_kh_cnt_arr[dist_arr <= max_ham_dist])

        if debug:
            print(f"{i_trial= }")

        max_hamball_ind = np.argmax(hamball_cnt_arr)
        consensus_kh = uniq_kh_arr[top_k_inds[max_hamball_ind]]
        hamball_proportion = (hamball_cnt_arr[max_hamball_ind] + 0.0) / n_total_kmer
        hamball_ratio = hamball_proportion / p_unif

        if hamball_ratio > ratio_cutoff:
            res_consensus_kh_list.append(consensus_kh)
            res_consensus_proportion_list.append(hamball_proportion)
            res_consensus_ratio_list.append(hamball_ratio)
            res_consensus_log10_pvalue_list.append( norm.logsf(hamball_ratio, loc=ratio_mu, scale=ratio_std)/np.log(10) )

            if merge_revcom_mode:
                rc_consensus_kh = revcom_hash(consensus_kh, kmer_len)
                seq_np_arr = mask_input(seq_np_arr, kmer_len, np.array([consensus_kh, rc_consensus_kh]), np.array([max_ham_dist, max_ham_dist]))
            else:
                seq_np_arr = mask_input(seq_np_arr, kmer_len, np.array([consensus_kh]), np.array([max_ham_dist]))

            hash_arr = comp_kmer_hash_taichi(seq_np_arr, kmer_len)
            uniq_kh_arr, uniq_kh_cnt_arr = count_uniq_hash(hash_arr, kmer_len)
            if merge_revcom_mode:
                uniq_kh_arr, uniq_kh_cnt_arr = merge_revcom(uniq_kh_arr, uniq_kh_cnt_arr, kmer_len,
                                                            keep_lower_hash_flag=True)
        else:
            break
    return dict(zip(res_consensus_kh_list, zip(res_consensus_proportion_list, res_consensus_ratio_list, res_consensus_log10_pvalue_list)))


def _convert_to_block_mat(uniq_dist_mat: np.ndarray, block_size_arr: np.ndarray) -> np.ndarray:
    """
    convert each element of uniq_dist_mat (square matrix) to a block, with all elements in the block having the same value
    Args:
        uniq_dist_mat: a nq x nq matrix
        block_size_arr: block size for each element in the output
    Returns:
        an expanded matrix
    """

    assert np.issubdtype(block_size_arr.dtype, np.integer)
    assert np.all(block_size_arr > 0)

    # expand the matrix
    boarder_arr = np.zeros(len(block_size_arr) + 1, dtype=int)
    boarder_arr[1:] = np.cumsum(block_size_arr)
    st_arr = boarder_arr[:-1]
    en_arr = boarder_arr[1:]
    n_seq = boarder_arr[-1]
    hamdist_mat = np.zeros((n_seq, n_seq), dtype=uniq_dist_mat.dtype)
    for i in range(len(block_size_arr)):
        for j in range(len(block_size_arr)):
            st_i, en_i = st_arr[i], en_arr[i]
            st_j, en_j = st_arr[j], en_arr[j]
            hamdist_mat[st_i:en_i, st_j:en_j] = uniq_dist_mat[i, j]
    return hamdist_mat


def _convert_to_block_arr(arr: np.ndarray, block_size_arr: np.ndarray) -> np.ndarray:
    """
    convert each element of arr to a block, with all elements in the block having the same value
    Args:
        arr: 1 x n array
        block_size_arr: block size for each element in the output, 1 x n array
    Returns:
        an expanded array
    """

    assert np.issubdtype(block_size_arr.dtype, np.integer)
    assert np.all(block_size_arr > 0)
    assert len(arr) == len(block_size_arr)

    # expand the matrix
    boarder_arr = np.zeros(len(block_size_arr) + 1, dtype=int)
    boarder_arr[1:] = np.cumsum(block_size_arr)
    st_arr = boarder_arr[:-1]
    en_arr = boarder_arr[1:]
    n_seq = boarder_arr[-1]
    out_arr = np.zeros(n_seq, dtype=arr.dtype)
    for i in range(len(block_size_arr)):
        st_i, en_i = st_arr[i], en_arr[i]
        out_arr[st_i:en_i] = arr[i]
    return out_arr

def cal_samp_kmer_hamdist_mat(samp_kh_arr: np.ndarray, samp_cnts: np.ndarray,
                           samp_label_arr: np.ndarray, conseq_list: List[str], kmer_len: int, uniq_dist_flag=False) -> np.ndarray:
    """
    Calculate the hamming distance between the sampled kmer
    When calculating distances for kmers with the same label, we only consider the first n letters, where n=len(conseq)
    Each uniq kmer is expanded uniq_kmer_cnt times
    Args:
        samp_kh_arr: sampled unique kmer hash arr, note that the kmers has been revcom-ed to align with the conseqs
        samp_cnts: counts of sampled kmer hash arr
        samp_label_arr: label of kmer, which conseq it belongs to
        conseq_list: consensus sequence list
        kmer_len: kmer length of the main conseqs
        uniq_dist_flag: if return the distance matrix for samp_kh_arr (unique values),
                        or expand it such that there are samp_cnts[i] replicates for samp_kh_arr[i]
    Returns:
        a hamming distance matrix, with each row representing a kmer
    """

    assert len(samp_kh_arr) == len(np.unique(samp_kh_arr)) # sample_kh_arr must only contain unique values
    n_uniq_kmer = len(samp_kh_arr)
    uniq_dist_mat = np.zeros((n_uniq_kmer, n_uniq_kmer), dtype=int)

    for conseq in conseq_list:
        assert len(conseq) <= kmer_len

    # calculate hamming dist based on kmer_len for all kmers
    for i, kh in enumerate(samp_kh_arr):
        uniq_dist_mat[i, (i+1):] = cal_hamming_dist(samp_kh_arr[(i+1):], kh, kmer_len)
        uniq_dist_mat[(i + 1):, i] = uniq_dist_mat[i, (i+1):]

    # calculate hamming dist for kmers belong to a short conseq
    for i, conseq in enumerate(conseq_list):
        if len(conseq) == kmer_len:
            continue
        tmpinds = np.where(samp_label_arr == i)[0]
        tmp_kh_arr = samp_kh_arr[tmpinds]
        conseq_len = len(conseq)
        tmp_kh_arr = np.right_shift(tmp_kh_arr, 2 * (kmer_len - conseq_len)).astype(get_hash_dtype(conseq_len))
        for j, kh in enumerate(tmp_kh_arr):
            tmp_dist_arr = cal_hamming_dist(tmp_kh_arr[(j + 1):], kh, conseq_len)
            uniq_dist_mat[tmpinds[j], tmpinds[(j + 1):]] = tmp_dist_arr
            uniq_dist_mat[tmpinds[(j + 1):], tmpinds[j]] = tmp_dist_arr

    # expand the matrix
    hamdist_mat = _convert_to_block_mat(uniq_dist_mat, samp_cnts)

    if uniq_dist_flag:
        return uniq_dist_mat
    else:
        return hamdist_mat


# sample display kmers
def sample_disp_kmer(conseq_list: List[str], kmer_len: int, motif_def_dict: dict, kmer_count_dir: Path,
                     n_total_sample=5000, n_motif_kmer=2500, revcom_mode=True) -> Tuple:
    """
    Sample kmers for visualization
    Args:
        conseq_list: consensus sequence list
        kmer_len: kmer length
        motif_def_dict: motif definition tables
        kmer_count_dir: output directory that contains the kmer count result
        n_total_sample: total number of samples for visualization
        n_motif_kmer: number of motif kmers, n_sample-n_motif_kmer is the number of random kmers
        revcom_mode: if reverse complement exist in data

    Returns:
        kmer_hash_arr, kmer_hash_cnt_arr, label_arr, conseq_list
    """
    # process conseq
    conseq_list = [s for s in conseq_list if 2 < len(s) <= kmer_len]
    assert len(conseq_list) > 0
    assert all([len(s_i) >= len(s_i_plus_1) for s_i, s_i_plus_1 in zip(conseq_list, conseq_list[1:])])
    #conseq_list = sorted(conseq_list, key=lambda x: len(x)) # sort conseq by length

    # load kmer counts
    kmer_cnt_file = kmer_count_dir / f"k{kmer_len}.pkl"  # [kmer_len, uniq_kh_arr, uniq_kh_cnt_arr]
    with open(kmer_cnt_file, "rb") as fh:
        res_list = pickle.load(fh)
    assert res_list[0] == kmer_len
    uniq_kh_arr, uniq_kh_cnt_arr = res_list[1], res_list[2]

    # ensure there are enough seqs to sample,
    sampling_flag = True
    if n_total_sample > sum(uniq_kh_cnt_arr):
        warnings.warn(f"The number of samples n_sample={n_total_sample} is larger than the original " +
                      f"data n_seq={sum(uniq_kh_cnt_arr)}, process and return original data.")
        sampling_flag = False

    # calculate hamming distance to each consensus
    n_conseq = len(conseq_list)
    n_uniq_kmer = len(uniq_kh_arr)
    ham_dist_mat = np.zeros((n_conseq, n_uniq_kmer), dtype=int)
    rc_flag_mat = np.zeros((n_conseq, n_uniq_kmer), dtype=bool) # if the min ham dist is from reverse complement
    for i, conseq in enumerate(conseq_list):
        conseq_kh = kmer2hash(conseq)
        dist_arr = cal_hamming_dist_head(uniq_kh_arr, conseq_kh, kmer_len, len(conseq))
        if revcom_mode:
            rc_conseq_kh = revcom_hash(conseq_kh, len(conseq))
            assert conseq_kh <= rc_conseq_kh
            rc_dist_arr = cal_hamming_dist_tail(uniq_kh_arr, rc_conseq_kh, kmer_len, len(conseq))  # revcom
            rc_flag_mat[i] = rc_dist_arr < dist_arr
            dist_arr = np.minimum(dist_arr, rc_dist_arr)
        ham_dist_mat[i] = dist_arr

    # label noise kmers for each consensus
    for i, conseq in enumerate(conseq_list):
        tmp_max_ham_dist = motif_def_dict[len(conseq)].max_ham_dist
        ham_dist_mat[i][ham_dist_mat[i] > tmp_max_ham_dist] = kmer_len  # maximum distance is kmer_len, so noise

    # assign label
    min_dist_arr = np.min(ham_dist_mat, axis=0)
    min_dist_ind_arr = np.argmin(ham_dist_mat, axis=0)
    min_dist_ind_arr[min_dist_arr > motif_def_dict[kmer_len].max_ham_dist] = n_conseq
    label_arr = min_dist_ind_arr

    # # assign label
    # min_dist_arr = np.min(ham_dist_mat, axis=0)
    # min_dist_ind_arr = np.argmin(ham_dist_mat, axis=0)
    # min_dist_ind_arr[min_dist_arr > motif_def_dict[kmer_len].max_ham_dist] = n_conseq
    # for i, conseq in enumerate(conseq_list):
    #     if len(conseq) < kmer_len:
    #         tmpinds = np.where(min_dist_ind_arr == i)[0]
    #         tmpinds = tmpinds[ min_dist_arr[tmpinds] > motif_def_dict[len(conseq)].max_ham_dist ]
    #         min_dist_ind_arr[tmpinds] = n_conseq
    # label_arr = min_dist_ind_arr # each conseq has a label, random seq corresponds to n_conseq

    # update kmer hash list for revcom mode, such that all the kmers in a hamming ball align with the consensus sequence
    if revcom_mode:
        for i, conseq in enumerate(conseq_list):
            tmpinds = np.where(label_arr == i)[0]
            for j in np.where(rc_flag_mat[i][tmpinds])[0]:
                tmpind = tmpinds[j]
                uniq_kh_arr[tmpind] = revcom_hash(uniq_kh_arr[tmpind], kmer_len)

    # no sampling case
    if not sampling_flag:
        return uniq_kh_arr, uniq_kh_cnt_arr, label_arr, conseq_list

    sample_cnt_arr = np.bincount(label_arr, weights=uniq_kh_cnt_arr)
    motif_weights = sample_cnt_arr[:-1] / sum(sample_cnt_arr[:-1])
    sample_cnt_arr[:-1] = np.around(n_motif_kmer * motif_weights)
    sample_cnt_arr[-1] = n_total_sample - sum(sample_cnt_arr[0:-1])
    sample_cnt_arr = sample_cnt_arr.astype(int)
    assert len(sample_cnt_arr) == n_conseq + 1

    # sampling
    samp_inds = []
    samp_cnts = []
    for c in range(n_conseq+1):
        c_inds = np.where(label_arr == c)[0]
        ws = uniq_kh_cnt_arr[c_inds]
        ws = ws/sum(ws)
        tmpcnts = np.random.multinomial(sample_cnt_arr[c], ws, size=1).squeeze()
        samp_inds.append(c_inds[tmpcnts > 0])
        samp_cnts.append(tmpcnts[tmpcnts > 0])

    samp_inds = np.concatenate(samp_inds)
    samp_cnts = np.concatenate(samp_cnts)
    samp_kh_arr = uniq_kh_arr[samp_inds]
    samp_label_arr = label_arr[samp_inds]

    return samp_kh_arr, samp_cnts, samp_label_arr, conseq_list # output pkl, txt file


def ex_hamball_kh_arr(res_dir: str, conseq: str, max_ham_dist: int=-1, motif_def_file: str=None, revcom_mode=True):
    """
    Extract kmer hash for all kmers of the Hamming ball centered on the input consensus sequence 
    Args:
        res_dir: result directory
        conseq: consensus sequence
        max_ham_dist: maximum Hamming distance of the Hamming ball
        motif_def_file: motif definition file
        revcom_mode: reverse complment mode
    Returns:
        tuple (unique kmer hash, unique kmer cnt)
    """
    conseq = conseq.upper()
    assert all([e in ("A", "C", "G", "T") for e in conseq])
    kmer_len = len(conseq)
    conseq_kh = kmer2hash(conseq)
    rc_conseq_kh = revcom_hash(conseq_kh, kmer_len)
    assert conseq_kh <= rc_conseq_kh

    assert Path(motif_def_file).exists()
    assert Path(res_dir).exists()
    res_path = Path(res_dir)

    # load kmer counts
    kmer_cnt_file = res_path / FileNameDict["kmer_count_dir"] / f"k{kmer_len}.pkl"  # [kmer_len, uniq_kh_arr, uniq_kh_cnt_arr]
    with open(kmer_cnt_file, "rb") as fh:
        res_list = pickle.load(fh)
    assert res_list[0] == kmer_len
    uniq_kh_arr, uniq_kh_cnt_arr = res_list[1], res_list[2]

    if max_ham_dist == -1:
        motif_def_dict = init_motif_def_dict(motif_def_file)
        max_ham_dist = motif_def_dict[kmer_len].max_ham_dist

    n_uniq_kmer = len(uniq_kh_arr)
    rc_flag_arr = np.zeros(n_uniq_kmer, dtype=bool)  # if the min ham dist is from reverse complement
    dist_arr = cal_hamming_dist(uniq_kh_arr, conseq_kh, kmer_len)
    if revcom_mode:
        rc_dist_arr = cal_hamming_dist(uniq_kh_arr, rc_conseq_kh, kmer_len)
        rc_flag_arr = rc_dist_arr < dist_arr
        dist_arr = np.minimum(dist_arr, rc_dist_arr)

    hamball_flag_arr = dist_arr <= max_ham_dist
    # update kmer hash list for revcom mode, such that all the kmers in a hamming ball align with the consensus sequence
    if revcom_mode:
        tmp_flag_arr = np.logical_and(rc_flag_arr, hamball_flag_arr)
        tmpinds = np.where(tmp_flag_arr)[0]
        for i in tmpinds:
            uniq_kh_arr[i] = revcom_hash(uniq_kh_arr[i], kmer_len)

    return uniq_kh_arr[hamball_flag_arr], uniq_kh_cnt_arr[hamball_flag_arr]


def cal_cnt_mat(uniq_kh_arr, uniq_kh_cnt_arr, kmer_len):
    cnt_mat = np.zeros((4, kmer_len), dtype=int)

    for kh, cnt in zip(uniq_kh_arr, uniq_kh_cnt_arr):
        kmer = hash2kmer(kh, kmer_len)
        kmer_arr = dna2arr(kmer, dtype=np.uint8, append_missing_val_flag=False)
        for i, b in enumerate(kmer_arr):
            cnt_mat[b][i] += cnt
    return cnt_mat


def _draw_logo(cnt_mat_numpy_file: str, output_fig_file=None):
    cntmat = np.loadtxt(cnt_mat_numpy_file, delimiter=",")
    cntmat = np.transpose(cntmat)
    n_pos = len(cntmat)
    cnt_df = pd.DataFrame(data=cntmat, index=np.arange(n_pos), columns=["A", "C", "G", "T"])
    cnt_df_info = logomaker.transform_matrix(cnt_df, from_type="counts", to_type="information")
    crp_logo = logomaker.Logo(cnt_df_info, font_name='Arial')
    if output_fig_file:
        plt.savefig(output_fig_file)



def gen_motif_co_occurence_mat(input_fasta_file: str, conseq_list: List[str], motif_def_dict: dict, min_motif_cnt=1, revcom_mode=True):
    """
    get the co-occurence count matrix of different motifs by scanning each read in the input fasta file
    Args:
        input_fasta_file: input fasta file
        conseq_list: consensus sequences of motifs
        motif_def_dict: motif definition table
        min_motif_cnt: minimum number of observed motif in each read
        revcom_mode: if reverse complement should be considered

    Returns:
        n_motif x n_motif count matrix

    """
    def read_stream(fh):
        for rec in SeqIO.parse(fh, "fasta"):
            yield str(rec.seq).upper()

    def read_fasta(input_fasta_file):
        if input_fasta_file.endswith(".gz"):
            with gzip.open(input_fasta_file, "rt") as fh:
                yield from read_stream(fh)
        else:
            with open(input_fasta_file, "r") as fh:
                yield from read_stream(fh)

    n_conseq = len(conseq_list)
    assert  n_conseq > 0
    res_mat = np.zeros((n_conseq, n_conseq), dtype=int)
    for seq in read_fasta(input_fasta_file):
        cnt_arr = get_motif_co_occurence_cnt_arr(seq, conseq_list, motif_def_dict, revcom_mode=revcom_mode)
        motif_inds = np.where(cnt_arr >= min_motif_cnt)[0]
        for i in range(len(motif_inds)):
            for j in range(i, len(motif_inds)):
                res_mat[motif_inds[i], motif_inds[j]] += 1
    diag_vec = np.diag(res_mat)
    diag_inds = np.diag_indices(n_conseq)
    res_mat += np.transpose(res_mat)
    res_mat[diag_inds] = diag_vec

    return res_mat


def get_motif_co_occurence_cnt_arr(dna_seq: str, conseq_list: List[str], motif_def_dict: dict, revcom_mode=True) -> np.ndarray:
    """
    get the counts of each input motif (hamming ball) for a given DNA sequence
    Args:
        dna_seq: input dna sequence
        conseq_list: consensus sequence list, consensus length can be different
        motif_def_dict: motif definition dictionary, kmer_len : MotifDef obj
        revcom_mode: if reverse complement should be considered
    Returns:
        a numpy array showing the counts for each motif
    """

    assert len(conseq_list) > 0
    conseq_kh_list = []
    conseq_len_list = []
    for conseq in conseq_list:
        conseq_len_list.append(len(conseq))
        conseq_kh_list.append(kmer2hash(conseq))

    seq_np_arr = dna2arr(dna_seq, append_missing_val_flag=False)
    kh_arr_dict = {}
    uniq_conseq_len_arr = np.unique(conseq_len_list)
    for kmer_len in uniq_conseq_len_arr:
        hash_arr = comp_kmer_hash_taichi(seq_np_arr, kmer_len)
        uniq_kh_arr, uniq_kh_cnt_arr = count_uniq_hash(hash_arr, kmer_len)
        kh_arr_dict[kmer_len] = (uniq_kh_arr, uniq_kh_cnt_arr)

    res = np.zeros(len(conseq_list), dtype=int)
    for i in range(len(conseq_list)):
        conseq_kh, conseq_len = conseq_kh_list[i], conseq_len_list[i]
        uniq_kh_arr, uniq_kh_cnt_arr = kh_arr_dict[conseq_len]
        max_ham_dist = motif_def_dict[conseq_len].max_ham_dist
        res[i] = _get_motif_cnt(uniq_kh_arr, uniq_kh_cnt_arr, conseq_len, conseq_kh, max_ham_dist,
                                rev_com_mode=revcom_mode)
    return res


def _get_motif_cnt(uniq_kh_arr: np.array, uniq_kh_cnt_arr: np.array,
                               kmer_len: int, conseq_kh: np.uint64,
                               max_ham_dist: int, rev_com_mode=True) -> int:
    """
    get the count of motif from input kh_arr
    Args:
        uniq_kh_arr: unique kmer hash array
        uniq_kh_cnt_arr: count of each unique kmer hash
        kmer_len: kmer length
        conseq_kh: hash of consequence sequence
        max_ham_dist: maximum hamming distance of hamming ball
        rev_com_mode: if revcom should be considered
    Returns:
        number of kmers fall in the hamming ball
    """
    dist_arr = cal_hamming_dist(uniq_kh_arr, conseq_kh, kmer_len)
    if rev_com_mode:
        rc_kh = revcom_hash(conseq_kh, kmer_len)
        rc_dist_arr = cal_hamming_dist(uniq_kh_arr, rc_kh, kmer_len)  # revcom
        dist_arr = np.minimum(dist_arr, rc_dist_arr)
    return int(np.sum(uniq_kh_cnt_arr[dist_arr <= max_ham_dist]))
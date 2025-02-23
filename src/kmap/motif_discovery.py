import matplotlib.pyplot as plt

from .kmer_count import (comp_kmer_hash_taichi, count_uniq_hash, merge_revcom,
                                 cal_hamming_dist, revcom_hash, mask_input, proc_input,
                                 init_motif_def_dict, mask_ham_ball, hash2kmer, kmer2hash,
                                 cal_hamming_dist_head, cal_hamming_dist_tail, dna2arr,
                                 reverse_complement, get_hash_dtype, FileNameDict,
                                 gen_motif_def_dict, get_invalid_hash, remove_duplicate_hash_per_seq)
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
        read_boarder_mat, seq_np_arr = pickle.load(fh)

    # mask user provided noise sequences from the input sequence
    if len(mask_noise_seq_list) > 0:
        max_ham_dist_list = [0 for _ in mask_noise_seq_list]
        seq_np_arr = mask_ham_ball(seq_np_arr, motif_def_dict, mask_noise_seq_list, max_ham_dist_list)

    # load necessary parameters
    top_k = config_dict["motif_discovery"]["top_k"]
    n_trial = config_dict["motif_discovery"]["n_trial"]
    orig_seq_np_arr = seq_np_arr.copy()

    background_mode = False
    if config_dict["general"]["background_fasta_file"] != "None":
        background_mode = True
        processed_background_fasta_file_path = Path(res_dir) / FileNameDict["processed_background_fasta_file"]
        assert processed_background_fasta_file_path.exists()
        with open(processed_background_fasta_file_path, "rb") as fh:
            background_read_boarder_mat, background_seq_np_arr = pickle.load(fh)

    kmer_count_dir = Path(res_dir) / FileNameDict["kmer_count_dir"]
    if not kmer_count_dir.exists():
        kmer_count_dir.mkdir()
    for kmer_len in range(min_k, max_k + 1):
        kmer_cnt_pkl_file = Path(kmer_count_dir) / get_kmer_count_pkl_file_name(kmer_len, "case")
        if not kmer_cnt_pkl_file.exists():
            hash_arr = comp_kmer_hash_taichi(seq_np_arr, kmer_len)
            uniq_kh_arr, uniq_kh_cnt_arr = count_uniq_hash(hash_arr, kmer_len, True)
            with open(kmer_cnt_pkl_file, "wb") as fh:
                pickle.dump([kmer_len, uniq_kh_arr, uniq_kh_cnt_arr], fh)

    # candidate motif sequences
    candidate_conseq_file = Path(res_dir) / FileNameDict["candidate_conseq_file"]
    if candidate_conseq_file.exists():
        print(f"{candidate_conseq_file} already exist, re-use it.")
    else:
        # motif discovery
        res = {}
        for kmer_len in range(min_k, max_k + 1):
            seq_np_arr[:] = orig_seq_np_arr[:]
            
            if background_mode:
                min_proportion = config_dict["motif_discovery"]["min_proportion"]
                min_ratio = config_dict["motif_discovery"]["min_ratio"]
                ratio_mu = motif_def_dict[kmer_len].ratio_mu
                ratio_std = motif_def_dict[kmer_len].ratio_std
                max_ham_dist = motif_def_dict[kmer_len].max_ham_dist

                consensus_kh_dict = find_motif_with_background(read_boarder_mat, seq_np_arr, 
                               background_read_boarder_mat, background_seq_np_arr, 
                               kmer_len, max_ham_dist,
                            min_proportion=min_proportion, 
                            min_ratio=min_ratio, 
                            ratio_mu=ratio_mu,
                            ratio_std=ratio_std,
                            top_k=top_k, n_trial=n_trial,
                            merge_revcom_mode=revcom_mode, 
                            debug=debug) # snp_np_arr are mutated in find_motif_with_background()

            else:
                p_uniform_k = motif_def_dict[kmer_len].p_uniform
                max_ham_dist = motif_def_dict[kmer_len].max_ham_dist
                ratio_mu = motif_def_dict[kmer_len].ratio_mu
                ratio_std = motif_def_dict[kmer_len].ratio_std
                ratio_cutoff = motif_def_dict[kmer_len].ratio_cutoff

                # (kmer_len, candiate_consensus_kh) as the key
                consensus_kh_dict = find_motif(read_boarder_mat, seq_np_arr, kmer_len, 
                                            max_ham_dist, p_uniform_k,
                                            ratio_mu, ratio_std, ratio_cutoff,
                                            top_k, n_trial,
                                            revcom_mode,
                                            debug=debug)  # snp_np_arr are mutated in find_motif()
                
            res.update(consensus_kh_dict)
            if debug:
                print(f"filtered consensus kmers when k = {kmer_len}")
            
        print(f"kmer counting finished for k={min_k}...{max_k}. Candidate consensus sequences generated.")

        print_lines_list = proc_find_motif_results(res)
        write_lines(print_lines_list, candidate_conseq_file)
    
    # read candidate consensus sequences from file
    candidate_conseq_file_lines = candidate_conseq_file.read_text().splitlines()
    file_header_fields = candidate_conseq_file_lines[0].split(",")
    conseq_idx = file_header_fields.index("kmer")
    motif_flag_idx = file_header_fields.index("candidate_motif_flag")
    full_candidate_conseq_list = [line.split(",")[conseq_idx] 
                                  for line in candidate_conseq_file_lines[1:]]
    candidate_conseq_list = [line.split(",")[conseq_idx] 
                             for line in candidate_conseq_file_lines[1:]
                             if line.split(",")[motif_flag_idx] == "True"]

    # merge candidate motif sequences
    final_conseq_file = Path(res_dir) / FileNameDict["final_conseq_file"]
    if final_conseq_file.exists():
        with open(final_conseq_file, "r") as fh:
            final_conseq_list = fh.read().splitlines()
        print(f"{final_conseq_file} already exist, re-use it.")
    else:
        final_conseq_list = merge_consensus_seqs(candidate_conseq_list)
        if len(final_conseq_list) == 0:
            raise ValueError("No consensus sequences found.")
        write_lines(final_conseq_list, final_conseq_file)

    final_conseq_info_file = Path(res_dir) / FileNameDict["final_conseq_info_file"]
    if final_conseq_info_file.exists():
        print(f"{final_conseq_info_file} already exist, re-use it.")
    else:
        final_conseq_info_list = ["ID," + candidate_conseq_file_lines[0]] # header
        for i,conseq in enumerate(final_conseq_list):
            conseq_idx = full_candidate_conseq_list.index(conseq)
            final_conseq_info_list.append(str(i) + "," + candidate_conseq_file_lines[conseq_idx+1])
        write_lines(final_conseq_info_list, final_conseq_info_file)
    print("Final consensus sequences generated.")

    if config_dict["motif_discovery"]["motif_pos_density_flag"]:
        # output the motif kmer position distribution
        x_step = 0.01
        x_arr = np.arange(0, 1.0 + x_step, x_step)
        res = []
        n_motif_seq_arr = []
        out_fig_dir = Path(res_dir) / FileNameDict["motif_pos_density_plot_dir"]
        if not out_fig_dir.exists():
            out_fig_dir.mkdir()
        for i, conseq in enumerate(final_conseq_list):
            # unormalized density, sum approx equal to the number of input sequences that have the motif
            n_motif_seq, density_arr = get_motif_position_distribution(res_dir, conseq, x_step=x_step, x_arr=x_arr, debug=debug)
            n_motif_seq_arr.append(n_motif_seq)
            out_fig_path = out_fig_dir / f"motif{i}-pos.pdf"
            title_str = f"motif {i}: {conseq} n_motif_seq={n_motif_seq}"
            _draw_motif_pos_density(title_str, x_arr, density_arr, out_fig_path)
            res.append(density_arr)
        res_mat = np.vstack(res)
        out_fig_path = out_fig_dir / f"motif_all_pos.pdf"
        _draw_motif_pos_density_all(x_arr, res_mat, final_conseq_list, n_motif_seq_arr, out_fig_path)
        out_pkl_file_path = Path(res_dir) / FileNameDict["motif_pos_density_file"]
        with open(out_pkl_file_path, "wb") as fh:
            pickle.dump([x_arr, res_mat], fh)
        print("motif position distribution generated.")

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
        co_sum_mat = np.diag(co_occur_mat) + np.diag(co_occur_mat).reshape((-1, 1))
        co_occur_norm_mat = co_occur_mat / (co_sum_mat - co_occur_mat)
        with open(co_occur_mat_file, "w+") as fh:
            np.savetxt(fh, co_occur_mat, delimiter="\t", fmt="%d")
        with open(co_occur_mat_norm_file, "w+") as fh:
            np.savetxt(fh, co_occur_norm_mat, delimiter="\t", fmt="%2.3f")
    print("motif co-occurence matrix generated.")

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


def find_motif(read_boarder_mat, seq_np_arr, kmer_len: int, max_ham_dist, p_unif,
               ratio_mu, ratio_std, ratio_cutoff, top_k=5, n_trial=10,
               merge_revcom_mode=True, debug=False) -> dict:
    """
    main motif discovery code,
        step 0: we try to pick the largest hamming ball for top k kmers each time,
        step 1: check if it passes the significant test
        step 2: repeat the process n_trial times
    Args:
        read_boarder_mat: read boarder matrix, n_seq x 2, each row is the start and end index of a read
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

    # first round
    hash_arr = comp_kmer_hash_taichi(seq_np_arr, kmer_len)
    n_reads = len(read_boarder_mat) # n_seq x 2

    uniq_kh_arr, uniq_kh_cnt_arr = count_uniq_hash(hash_arr, kmer_len, merge_revcom_mode)

    # count total kmer excluding invalid kmer
    n_total_kmer = sum(uniq_kh_cnt_arr)

    results = {}
    results["headers"] = ["candidate_motif_flag", "case_hamball_cnt", "case_total_kmer_cnt",
                           "control_hamball_cnt", "control_total_kmer_cnt", "size_factor_case_vs_control",
                           "hamball_ratio", "log10_pvalue",
                           "n_motif_reads_case",  "n_reads_case", "n_motif_reads_control", "n_reads_control", "under_represented_flag"]

    for i_trial in range(n_trial):
        # get the kmer with maximum hamming ball counts
        top_k_inds = np.array(np.argpartition(uniq_kh_cnt_arr, -top_k)[-top_k:])
        if len(top_k_inds) == 0:
            break

        hamball_cnt_arr = np.zeros(top_k)
        for i, ind in enumerate(top_k_inds):
            kh = uniq_kh_arr[ind]
            dist_arr = cal_hamming_dist(uniq_kh_arr, kh, kmer_len, merge_revcom_mode)
            hamball_cnt_arr[i] = np.sum(uniq_kh_cnt_arr[dist_arr <= max_ham_dist])

        if debug:
            print(f"{i_trial= }")

        max_hamball_ind = np.argmax(hamball_cnt_arr)
        consensus_kh = uniq_kh_arr[top_k_inds[max_hamball_ind]]
        hamball_proportion = (hamball_cnt_arr[max_hamball_ind] + 0.0) / n_total_kmer
        hamball_ratio = hamball_proportion / p_unif

        if hamball_ratio > ratio_cutoff:
            candidate_motif_flag = True
            n_motif_reads = get_hamball_read_count(read_boarder_mat, hash_arr, kmer_len,
                                                        consensus_kh, max_ham_dist, merge_revcom_mode)
            log10_pvalue = norm.logsf(hamball_ratio, loc=ratio_mu, scale=ratio_std)/np.log(10)
            results[(kmer_len, consensus_kh)] = (candidate_motif_flag, 
                                                hamball_cnt_arr[max_hamball_ind], n_total_kmer,
                                                np.nan, np.nan, np.nan,
                                                hamball_ratio, log10_pvalue,
                                                n_motif_reads, n_reads,
                                                np.nan, np.nan, np.nan)
            
            if merge_revcom_mode:
                rc_consensus_kh = revcom_hash(consensus_kh, kmer_len)
                seq_np_arr = mask_input(seq_np_arr, kmer_len, np.array([consensus_kh, rc_consensus_kh]), np.array([max_ham_dist, max_ham_dist]))
            else:
                seq_np_arr = mask_input(seq_np_arr, kmer_len, np.array([consensus_kh]), np.array([max_ham_dist]))

            hash_arr = comp_kmer_hash_taichi(seq_np_arr, kmer_len)
            uniq_kh_arr, uniq_kh_cnt_arr = count_uniq_hash(hash_arr, kmer_len, merge_revcom_mode)
            
        else:
            break
    return results


def find_motif_with_background(case_read_boarder_mat, case_seq_np_arr, 
                               control_read_boarder_mat, control_seq_np_arr, 
                               kmer_len: int, max_ham_dist: int,
                            min_proportion: float = 0.3, min_ratio: float = 1.2, 
                            top_k: int = 5, n_trial: int = 100, ratio_mu: float=1.0, ratio_std = 0.1,
                            merge_revcom_mode: bool = True, debug: bool = False) -> dict:
    """
    Main motif discovery code that compares case and control samples
    Args:
        case_seq_np_arr: case sample sequence array
        control_seq_np_arr: control sample sequence array
        kmer_len: kmer length
        max_ham_dist: maximum hamming ball distance
        min_proportion: minimum proportion of reads containing the motif (default 0.3)
        min_ratio: minimum ratio between case/control hamming ball counts (default 1.5)
        top_k: top k consensus sequences to consider
        n_trial: number of times we try to pick up a motif
        merge_revcom_mode: if revcom should be merged
        debug: print debug info
    Returns:
        dict: key is consensus_kmer_hash, value is tuple of (case_count, control_count, ratio)
    """
    # Get number of reads in case and control
    n_reads_case = len(case_read_boarder_mat)
    n_reads_control = len(control_read_boarder_mat)
    
    # First compute hashes and counts for case sample
    case_hash_arr = comp_kmer_hash_taichi(case_seq_np_arr, kmer_len)
    case_uniq_kh_arr, case_uniq_kh_cnt_arr = count_uniq_hash(case_hash_arr, kmer_len, merge_revcom_mode)
    case_total_kmer_cnt = np.sum(case_uniq_kh_cnt_arr)
         
    # then compute hashes and counts for control sample
    control_hash_arr = comp_kmer_hash_taichi(control_seq_np_arr, kmer_len)
    control_uniq_kh_arr, control_uniq_kh_cnt_arr = count_uniq_hash(control_hash_arr, kmer_len, merge_revcom_mode)
    control_total_kmer_cnt = np.sum(control_uniq_kh_cnt_arr) 
    size_factor_case_vs_control = case_total_kmer_cnt / (control_total_kmer_cnt + 1e-6)
    # Merge case and control kmer hashes and counts
    combined_uniq_kh_arr, combined_uniq_cnt_arr, case_idx_map, control_idx_map = merge_sorted_kh_arrays(
        case_uniq_kh_arr, case_uniq_kh_cnt_arr,
        control_uniq_kh_arr, control_uniq_kh_cnt_arr,
        size_factor_case_vs_control
    )

    # Store results
    results = {"headers": ["candidate_motif_flag", "case_hamball_cnt", "case_total_kmer_cnt",
                           "control_hamball_cnt", "control_total_kmer_cnt", "size_factor_case_vs_control",
                           "hamball_ratio", "log10_pvalue",
                           "n_motif_reads_case",  "n_reads_case", "n_motif_reads_control", "n_reads_control", "under_represented_flag"]}

    # Main discovery loop
    for i_trial in range(n_trial):
        if debug:
            print(f"Trial {i_trial}")

        # 1. Get top k kmers with highest counts
        top_k_inds = np.array(np.argpartition(combined_uniq_cnt_arr, -top_k)[-top_k:])
        if len(top_k_inds) == 0:
            break

        # 2. Calculate hamming ball counts for each top kmer
        hamball_cnt_arr = np.zeros(top_k)
        for i, ind in enumerate(top_k_inds):
            kh = combined_uniq_kh_arr[ind]
            dist_arr = cal_hamming_dist(combined_uniq_kh_arr, kh, kmer_len, merge_revcom_mode)
            hamball_cnt_arr[i] = np.sum(combined_uniq_cnt_arr[dist_arr <= max_ham_dist])

        # 3. Get kmer with maximum hamming ball count
        max_hamball_ind = np.argmax(hamball_cnt_arr)
        consensus_kh = combined_uniq_kh_arr[top_k_inds[max_hamball_ind]]

        # 4. break if the maximum hamming ball occurs in less than 30% reads in both case and control
        n_motif_reads_case = get_hamball_read_count(case_read_boarder_mat, case_hash_arr, kmer_len,
                                                        consensus_kh, max_ham_dist, merge_revcom_mode)
        n_motif_reads_control = get_hamball_read_count(control_read_boarder_mat, control_hash_arr, kmer_len,
                                                        consensus_kh, max_ham_dist, merge_revcom_mode)
        motif_proportion_case = n_motif_reads_case / n_reads_case
        motif_proportion_control = n_motif_reads_control / n_reads_control
        if motif_proportion_case < min_proportion and motif_proportion_control < min_proportion:
            if debug:
                print(f"Breaking: motif proportion too low - case: {motif_proportion_case:.3f}, control: {motif_proportion_control:.3f}")
            break

        # 5. Calculate hamming ball ratio between case and control
        dist_arr = cal_hamming_dist(combined_uniq_kh_arr, consensus_kh, kmer_len, merge_revcom_mode)
        hamball_flag_arr = dist_arr <= max_ham_dist
        
        case_hamball_cnt = np.sum(case_uniq_kh_cnt_arr[hamball_flag_arr[case_idx_map]])
        control_hamball_cnt = np.sum(control_uniq_kh_cnt_arr[hamball_flag_arr[control_idx_map]])
        
        # 6. check if the hamming ball ratio is larger than min_ratio
        hamball_ratio = case_hamball_cnt / (control_hamball_cnt + 1e-6) /  (size_factor_case_vs_control + 1e-6)
        # record but do not process under-represented kmers
        log10_pvalue = norm.logsf(hamball_ratio, loc=ratio_mu, scale=ratio_std)/np.log(10) 
        candidate_motif_flag = False
        under_represented_flag = False
        if hamball_ratio > min_ratio:
            candidate_motif_flag = True
        elif hamball_ratio < 1/min_ratio:
            under_represented_flag = True
        
        results[(kmer_len, consensus_kh)] = (candidate_motif_flag, 
                                            case_hamball_cnt, case_total_kmer_cnt, 
                                            control_hamball_cnt, control_total_kmer_cnt, size_factor_case_vs_control, 
                                            hamball_ratio, log10_pvalue, 
                                            n_motif_reads_case, n_reads_case,
                                            n_motif_reads_control, n_reads_control, under_represented_flag)

        # 7. Mask the hamming ball in case & control sequence for next iteration
        if merge_revcom_mode:
            rc_consensus_kh = revcom_hash(consensus_kh, kmer_len)
            case_seq_np_arr = mask_input(case_seq_np_arr, kmer_len, 
                                       np.array([consensus_kh, rc_consensus_kh]), 
                                       np.array([max_ham_dist, max_ham_dist]))
            control_seq_np_arr = mask_input(control_seq_np_arr, kmer_len,
                                       np.array([consensus_kh, rc_consensus_kh]),
                                       np.array([max_ham_dist, max_ham_dist]))  
        else:
            case_seq_np_arr = mask_input(case_seq_np_arr, kmer_len,
                                       np.array([consensus_kh]),
                                       np.array([max_ham_dist]))
            control_seq_np_arr = mask_input(control_seq_np_arr, kmer_len,
                                       np.array([consensus_kh]),
                                       np.array([max_ham_dist]))

        # 8. Recompute case hashes and counts for next iteration
        case_hash_arr = comp_kmer_hash_taichi(case_seq_np_arr, kmer_len)
        control_hash_arr = comp_kmer_hash_taichi(control_seq_np_arr, kmer_len)
        case_uniq_kh_arr, case_uniq_kh_cnt_arr = count_uniq_hash(case_hash_arr, kmer_len, merge_revcom_mode)
        control_uniq_kh_arr, control_uniq_kh_cnt_arr = count_uniq_hash(control_hash_arr, kmer_len, merge_revcom_mode)
        
        # Merge case and control kmer hashes and counts
        combined_uniq_kh_arr, combined_uniq_cnt_arr, case_idx_map, control_idx_map = merge_sorted_kh_arrays(
                case_uniq_kh_arr, case_uniq_kh_cnt_arr,
                control_uniq_kh_arr, control_uniq_kh_cnt_arr,
                size_factor_case_vs_control) 

    return results


def proc_find_motif_results(results: dict) -> List[str]:
    output_headers = ["kmer_len", "kmer_hash", "rc_kmer_hash", "kmer", "rc_kmer", "candidate_motif_flag", "under_represented_flag",
                           "case_hamball_cnt", "case_total_kmer_cnt",
                           "control_hamball_cnt", "control_total_kmer_cnt", "size_factor_case_vs_control",
                           "hamball_ratio", "log2_hamball_ratio", "log10_pvalue",
                           "n_motif_reads_case",  "n_reads_case", "motif_proportion_case", 
                           "n_motif_reads_control", "n_reads_control", "motif_proportion_control"]
    print_str_list = [",".join(output_headers)]
    headers = results["headers"]
    for key, value in results.items():
        if type(key)==str:
            continue
        kmer_len, kmer_hash = key
        tmp_dict = {hed:val for hed, val in zip(headers, value)}
        tmp_dict["kmer_len"] = kmer_len
        tmp_dict["log2_hamball_ratio"] = f"{np.log2(tmp_dict['hamball_ratio']):.3f}" 
        tmp_dict["motif_proportion_case"] = f"{tmp_dict['n_motif_reads_case'] / tmp_dict['n_reads_case']:.3f}"
        tmp_dict["motif_proportion_control"] = f"{tmp_dict['n_motif_reads_control'] / tmp_dict['n_reads_control']:.3f}"
        tmp_dict["kmer_hash"] = kmer_hash
        tmp_dict["rc_kmer_hash"] = revcom_hash(kmer_hash, kmer_len)
        tmp_dict["kmer"] = hash2kmer(kmer_hash, kmer_len)
        tmp_dict["rc_kmer"] = hash2kmer(revcom_hash(kmer_hash, kmer_len), kmer_len)
        tmp_list = [str(tmp_dict[hed]) for hed in output_headers]
        print_str_list.append(",".join(tmp_list))
    return print_str_list


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
        uniq_dist_mat[i, (i+1):] = cal_hamming_dist(samp_kh_arr[(i+1):], kh, kmer_len, False)
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
            tmp_dist_arr = cal_hamming_dist(tmp_kh_arr[(j + 1):], kh, conseq_len, False)
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
    kmer_cnt_file = kmer_count_dir / get_kmer_count_pkl_file_name(kmer_len, "case") # [kmer_len, uniq_kh_arr, uniq_kh_cnt_arr]
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
    #kmer_cnt_file = res_path / FileNameDict["kmer_count_dir"] / f"k{kmer_len}.pkl"  # [kmer_len, uniq_kh_arr, uniq_kh_cnt_arr]
    kmer_cnt_file = res_path / FileNameDict["kmer_count_dir"] /get_kmer_count_pkl_file_name(kmer_len, "case")
    with open(kmer_cnt_file, "rb") as fh:
        res_list = pickle.load(fh)
    assert res_list[0] == kmer_len
    uniq_kh_arr, uniq_kh_cnt_arr = res_list[1], res_list[2]

    if max_ham_dist == -1:
        motif_def_dict = init_motif_def_dict(motif_def_file)
        max_ham_dist = motif_def_dict[kmer_len].max_ham_dist

    n_uniq_kmer = len(uniq_kh_arr)
    rc_flag_arr = np.zeros(n_uniq_kmer, dtype=bool)  # if the min ham dist is from reverse complement
    dist_arr = cal_hamming_dist(uniq_kh_arr, conseq_kh, kmer_len, revcom_mode)

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


def _draw_motif_pos_density(title: str, x_arr: np.ndarray, y_arr: np.ndarray, out_fig_path: str|Path=None):
    plt.clf()
    plt.fill_between(x_arr, y_arr, alpha=0.5)
    plt.xlabel(f"relative motif position in sequence")
    plt.ylabel("density")
    plt.title(title)
    if out_fig_path:
        plt.savefig(out_fig_path)

def _draw_motif_pos_density_all(x_arr: np.ndarray, y_mat: np.ndarray, conseq_list: List[str],
                                n_motif_seq_arr: List, out_fig_path: str|Path=None):
    plt.clf()
    for i, conseq in enumerate(conseq_list):
        plt.plot(x_arr, y_mat[i], label=f"m{i}-{conseq} n={n_motif_seq_arr[i]}")
    plt.xlabel(f"relative motif position in sequence")
    plt.ylabel("density")
    plt.legend(loc="upper left")
    plt.title("motif position distribution")
    if out_fig_path:
        plt.savefig(out_fig_path)


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
        uniq_kh_arr, uniq_kh_cnt_arr = count_uniq_hash(hash_arr, kmer_len, False)
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
    dist_arr = cal_hamming_dist(uniq_kh_arr, conseq_kh, kmer_len, rev_com_mode)
    return int(np.sum(uniq_kh_cnt_arr[dist_arr <= max_ham_dist]))


def get_motif_position_distribution(res_dir: str, conseq: str, x_step=0.01, x_arr=None, debug=False):
    """
    get the position distribution of motif kmers on input sequences
    motif is firstly searched on the forward strand, if none is found, then search the reverse strand
    Args:
        res_dir: result directory
        conseq: consensus sequence
        x_step: step size of x in the returned density
        debug: debug mode
    Returns:
        n_motif_seq, position distribution of motif kmers on input sequences (kernel density, un-normalized)
    """

    config_file_name = FileNameDict["config_file"]  # config.toml
    config_file_path = Path(res_dir) / config_file_name

    motif_def_file = FileNameDict["motif_def_file"]  # motif_def_table.csv
    motif_def_file_path = Path(res_dir) / motif_def_file

    proc_fasta_file = FileNameDict["processed_fasta_file"]  # input.bin.pkl
    proc_fasta_file_path = Path(res_dir) / proc_fasta_file

    assert config_file_path.exists()
    assert motif_def_file_path.exists()
    assert proc_fasta_file_path.exists()

    # load config and motif_def files
    with open(config_file_path, "rb") as fh:
        config_dict = tomllib.load(fh)
    motif_def_dict = gen_motif_def_dict(config_dict, debug=debug)
    revcom_mode = config_dict["kmer_count"]["revcom_mode"]
    #rep_mode = config_dict["general"]["repetitive_mode"]

    with open(proc_fasta_file_path, "rb") as fh:
        read_boarder_mat, seq_np_arr = pickle.load(fh)

    kmer_len = len(conseq)
    hash_dtype = get_hash_dtype(kmer_len)
    invalid_hash = get_invalid_hash(hash_dtype)
    max_ham_dist = motif_def_dict[kmer_len].max_ham_dist

    conseq_kh = kmer2hash(conseq)
    rc_conseq_kh = revcom_hash(conseq_kh, kmer_len)
    hash_arr = comp_kmer_hash_taichi(seq_np_arr, kmer_len)
    dist_arr = cal_hamming_dist(hash_arr, conseq_kh, kmer_len, False)
    dist_arr[hash_arr == invalid_hash] = kmer_len
    motif_flag_arr = dist_arr <= max_ham_dist
    if revcom_mode:
        rc_dist_arr = cal_hamming_dist(hash_arr, rc_conseq_kh, kmer_len, False)  # revcom
        rc_dist_arr[hash_arr == invalid_hash] = kmer_len
        rc_motif_flag_arr = dist_arr <= max_ham_dist

    if x_arr is None:
        x_arr = np.arange(0, 1, x_step)
    density = np.zeros_like(x_arr)
    n_seq_forward, n_seq_reverse = 0, 0
    i_seq = 0
    for st, en in read_boarder_mat:
        # check if the forward strand has a motif
        if any(motif_flag_arr[st:en]):
            # if conseq_kh == 0: # for debug purposes
            #     tmphash_arr = hash_arr[st:en][motif_flag_arr[st:en]]
            #     print(f"{i_seq=}",[hash2kmer(kh, kmer_len) for kh in tmphash_arr])
            tmpinds = np.where(motif_flag_arr[st:en])[0]
            motif_rel_pos_arr = tmpinds / (en - st - kmer_len + 1)
            density += sum(norm(xi, scale=x_step).pdf(x_arr) for xi in motif_rel_pos_arr) / len(motif_rel_pos_arr)
            n_seq_forward += 1
        # check if the reverse strand has a motif
        elif revcom_mode and any(rc_motif_flag_arr[st:en]):
            tmpinds = np.where(rc_motif_flag_arr[st:en])[0]
            motif_rel_pos_arr = 1 - tmpinds / (en - st - kmer_len + 1)
            density += sum(norm(xi, scale=x_step).pdf(x_arr) for xi in motif_rel_pos_arr) / len(motif_rel_pos_arr)
            n_seq_reverse += 1
        i_seq += 1

    if debug:
        print(f"conseq={conseq} {n_seq_forward=} {n_seq_reverse=} "
              f"n_seq={n_seq_forward + n_seq_reverse} n_all_seq={len(read_boarder_mat)}")


    return n_seq_forward + n_seq_reverse, density


def merge_sorted_kh_arrays(kh_arr1: np.ndarray, cnt_arr1: np.ndarray, 
                          kh_arr2: np.ndarray, cnt_arr2: np.ndarray, 
                          size_factor: float=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge two sorted kmer hash arrays and their corresponding count arrays.
    If a kmer hash appears in both arrays, sum their counts.
    
    Args:
        kh_arr1: first sorted array of kmer hashes
        cnt_arr1: counts corresponding to kh_arr1
        kh_arr2: second sorted array of kmer hashes
        cnt_arr2: counts corresponding to kh_arr2
        
    Returns:
        Tuple of (merged_kh_arr, merged_cnt_arr, idx_map1, idx_map2) where:
        - merged_kh_arr: merged hash array
        - merged_cnt_arr: merged count array
        - idx_map1: indices mapping kh_arr1 to merged_kh_arr
        - idx_map2: indices mapping kh_arr2 to merged_kh_arr
    """
    # Pre-allocate maximum possible space
    n1, n2 = len(kh_arr1), len(kh_arr2)
    max_size = n1 + n2
    merged_kh_arr = np.empty(max_size, dtype=kh_arr1.dtype)
    merged_cnt_arr = np.empty(max_size, dtype=cnt_arr1.dtype)
    
    # Arrays to store mapping from original arrays to merged array
    idx_map1 = np.full(n1, -1, dtype=np.int64)  # -1 indicates no mapping
    idx_map2 = np.full(n2, -1, dtype=np.int64)
    
    # Initialize pointers and result index
    i, j, k = 0, 0, 0

    # Merge arrays while comparing elements
    while i < n1 and j < n2:
        if kh_arr1[i] < kh_arr2[j]:
            merged_kh_arr[k] = kh_arr1[i]
            merged_cnt_arr[k] = cnt_arr1[i]
            idx_map1[i] = k
            i += 1
        elif kh_arr1[i] > kh_arr2[j]:
            merged_kh_arr[k] = kh_arr2[j]
            merged_cnt_arr[k] = cnt_arr2[j] * size_factor
            idx_map2[j] = k
            j += 1
        else:  # Equal hashes - sum the counts
            merged_kh_arr[k] = kh_arr1[i]
            merged_cnt_arr[k] = cnt_arr1[i] + cnt_arr2[j] * size_factor
            idx_map1[i] = k
            idx_map2[j] = k
            i += 1
            j += 1
        k += 1
    
    # Add remaining elements from first array
    if i < n1:
        remain = n1 - i
        merged_kh_arr[k:k+remain] = kh_arr1[i:]
        merged_cnt_arr[k:k+remain] = cnt_arr1[i:]
        idx_map1[i:] = np.arange(k, k+remain)
        k += remain
    
    # Add remaining elements from second array
    if j < n2:
        remain = n2 - j
        merged_kh_arr[k:k+remain] = kh_arr2[j:]
        merged_cnt_arr[k:k+remain] = cnt_arr2[j:] * size_factor
        idx_map2[j:] = np.arange(k, k+remain)
        k += remain
    
    # Return arrays trimmed to actual size
    return merged_kh_arr[:k], merged_cnt_arr[:k], idx_map1, idx_map2

def get_hamball_read_count(read_boarder_mat: np.ndarray, hash_arr: np.ndarray, 
                           kmer_len: int, consensus_kh: np.uint64, max_ham_dist: int, merge_revcom_mode: bool):
    """
    get the number of reads that contain the motif
    Args:
        read_boarder_mat: read boarder matrix, n_seq x 2, each row is the start and end index of a read
        hash_arr: hash array computed from sequence numpy array
        consensus_kh: hash of the motif
        max_ham_dist: maximum hamming distance of the motif
        merge_revcom_mode: if reverse complement should be considered
    Returns:
        number of reads that contain the motif
    """
    dist_arr = cal_hamming_dist(hash_arr, consensus_kh, kmer_len, merge_revcom_mode)
    
    n_motif_reads = 0
    for st, en in read_boarder_mat:
        if any(dist_arr[st:en] <= max_ham_dist):
            n_motif_reads += 1
    return n_motif_reads


def get_kmer_count_pkl_file_name(kmer_len: int, label: str) -> str:
    return f"kmer_cnt_{kmer_len}_{label}.pkl"

import unittest
from src.kmap.kmer_count import *
from src.kmap.kmer_count import _preproc
from pathlib import Path

import pickle
import numpy as np


import os
import glob


class TestKmerCount(unittest.TestCase):
    def test__preproc(self):
        fasta_file = "./tests/test.fa"
        res_dir = "./test"

        # remove everything from output directory
        os.system(f'rm -rf {res_dir}')

        config_dict, motif_def_dict = _preproc(fasta_file, res_dir, rep_mode=True)
        boarder_file = Path(res_dir) / FileNameDict["processed_fasta_seqboarder_file"]
        self.assertTrue(boarder_file.exists())

        with open(boarder_file, "rb") as fh:
            boarder_mat = pickle.load(fh)

        for i, arr in enumerate(read_dnaseq_file(fasta_file)):
            self.assertEqual(len(arr), boarder_mat[i][1]-boarder_mat[i][0])

    def test_remove_duplicate_hash_per_seq(self):
        fasta_file = "./tests/test.fa"
        res_dir = "./test"

        # remove everything from output directory
        os.system(f'rm -rf {res_dir}')

        config_dict, motif_def_dict = _preproc(fasta_file, res_dir, rep_mode=False)
        boarder_file = Path(res_dir) / FileNameDict["processed_fasta_seqboarder_file"]
        self.assertTrue(boarder_file.exists())

        with open(boarder_file, "rb") as fh:
            boarder_mat = pickle.load(fh)

        seq_pkl_file = Path(res_dir) / FileNameDict["processed_fasta_file"]
        self.assertTrue(seq_pkl_file.exists())
        with open(seq_pkl_file, "rb") as fh:
            seq_np_arr = pickle.load(fh)

        kmer_len = 8
        hash_arr = comp_kmer_hash_taichi(seq_np_arr, kmer_len)
        hash_dtype = get_hash_dtype(kmer_len)
        invalid_hash = get_invalid_hash(hash_dtype)
        #print(f"{invalid_hash= } {invalid_hash: 0x}")
        hash_arr = remove_duplicate_hash_per_seq(hash_arr, boarder_mat, invalid_hash)

        # line 0 in test.fa, all A sequence
        i = 0
        all_A_seq_hash_arr = hash_arr[boarder_mat[i][0]:boarder_mat[i][1]]
        self.assertEqual(all_A_seq_hash_arr[0], kmer2hash("A" * kmer_len))
        for kh in all_A_seq_hash_arr[1:]:
            self.assertEqual(kh, invalid_hash)
        breakpoint()
        # line 1 in test.fa, CA-repeat sequence
        i = 1
        all_CA_seq_hash_arr = hash_arr[boarder_mat[i][0]:boarder_mat[i][1]]
        self.assertEqual(all_CA_seq_hash_arr[0], kmer2hash("CA" * int(kmer_len/2) ))
        self.assertEqual(all_CA_seq_hash_arr[1], kmer2hash("AC" * int(kmer_len / 2)))
        for kh in all_CA_seq_hash_arr[2:]:
            self.assertEqual(kh, invalid_hash)












from .inimotif import *
from src.kmap.kmer_count import *
from src.kmap.motif_discovery import *
from src.kmap.visualization import kmap, plot_2d_data, knn_smooth, _visualize_kmers
from src.kmap.motif_discovery import _convert_to_block_arr, _convert_to_block_mat, _scan_motif
from src.kmap.kmer_count import _preproc

import random

# TODO: TEST and parameter modification
def main(input_fasta_file, min_kmer_len, max_kmer_len,
         mask_consensus_list: List[str]=(),
         mask_noise_seq_list: List[str]=(),
         motif_def_file='.src/kmap/default_motif_def_table.csv',
         out_dir=".", debug=False):
    def write_csv(res, outfile):
        with open(outfile, 'w+') as fh:
            for line in res:
                fh.write(",".join(str(line)))

    motif_def_dict = init_motif_def_dict(motif_def_file)

    # preprocess: convert input fasta file into pkl file
    proc_input(input_fasta_file, out_dir)

    # load input pickle file as numpy array
    input_pkl_file = out_dir + f"/input.bin.pkl"
    with open(input_pkl_file, "rb") as fh:
        seq_np_arr = pickle.load(fh)

    # mask user provided noise sequnces from the input sequence
    if len(mask_noise_seq_list) > 0:
        max_ham_dist_list = [0 for _ in mask_noise_seq_list]
        seq_np_arr = mask_ham_ball(seq_np_arr, motif_def_dict, mask_noise_seq_list, max_ham_dist_list)

    # mask user provided consensus (Hamming ball) from the input sequence
    if len(mask_consensus_list) > 0:
        seq_np_arr = mask_ham_ball(seq_np_arr, motif_def_dict, mask_consensus_list)

    orig_seq_np_arr = seq_np_arr.copy()

    res = ["kmer_len,hash,kmer,proportion"]
    for kmer_len in range(min_kmer_len, max_kmer_len):
        seq_np_arr[:] = orig_seq_np_arr[:]
        p_uniform_k = motif_def_dict[kmer_len].p_uniform
        max_ham_dist = motif_def_dict[kmer_len].max_ham_dist
        p_cut_off = 1.3
        top_k = 5
        n_trial = 10
        revcom_mode = True

        assert min_kmer_len <= kmer_len <= max_kmer_len
        kmer_cnt_file = out_dir + f"/k{kmer_len}.pkl" # [kmer_len, uniq_kh_arr, uniq_kh_cnt_arr]
        consensus_kh_dict = find_motif(seq_np_arr, kmer_len, max_ham_dist, p_uniform_k, p_cut_off, top_k, n_trial,
                                       revcom_mode, rep_mode=True, save_kmer_cnt_flag=True,
                                       kmer_cnt_pkl_file=kmer_cnt_file, debug=debug)  # snp_np_arr are mutated in find_motif()

        print(f"filtered consensus kmers when k = {kmer_len}")
        for kh, prop in consensus_kh_dict.items():
            print(f'{hash2kmer(kh, kmer_len)} perc={prop * 100:0.5f}%')
            res.append(f"{kmer_len},{kh},{hash2kmer(kh, kmer_len)},{prop:0.8f}")

    outfile = out_dir + "/res_kmers.csv"
    write_csv(res, outfile)


def test_init_motif_def_dict():
    motif_def_dict = init_motif_def_dict("../src/kmap/default_motif_def_table.csv")
    print(motif_def_dict)
    assert len(motif_def_dict) == 18
    assert isinstance(motif_def_dict[7], MotifDef)


def gen_test_fa_file(conseq1="AATCGATAGC", conseq2="AGGACCTACGTAC", n_seq=1000, min_len=30, max_len=60, output_seq_file='./test/test.fa'):
    import random

    def generate_random_sequence(length):
        bases = ['A', 'C', 'G', 'T']
        return ''.join(random.choice(bases) for _ in range(length))

    def gen_motif_seq(conseq, length, mutation_rate=0.05):
        # 54% sequence are expected to be the same as conseq when mutation rate is 0.05, (1-0.05)**12=0.54
        conseq_len = len(conseq)
        bases = ['A', 'C', 'G', 'T']
        seq_base_list = [random.choice(bases) for _ in range(length)]
        pos = random.randint(0, length - conseq_len)
        for i, b in enumerate(conseq):
            if random.random() > mutation_rate:
                seq_base_list[pos + i] = b
        return "".join(seq_base_list)

    n_seq_1 = int(n_seq * 0.4)
    n_seq_2 = int(n_seq * 0.4)
    n_seq_other = n_seq - n_seq_1 - n_seq_2

    motif1_list = [gen_motif_seq(conseq1, random.randint(min_len, max_len)) for _ in range(n_seq_1)]
    motif2_list = [gen_motif_seq(conseq2, random.randint(min_len, max_len)) for _ in range(n_seq_2)]
    rand_list = [generate_random_sequence(random.randint(min_len, max_len)) for _ in range(n_seq_other)]

    with open(output_seq_file, 'w+') as fh:
        k = 0
        for s in motif1_list:
            fh.write(f'>seq{k}_1\n')
            fh.write(s + '\n')
            k += 1
        for s in motif2_list:
            fh.write(f'>seq{k}_2\n')
            fh.write(s + '\n')
            k += 1
        for s in rand_list:
            fh.write(f'>seq{k}_rand\n')
            fh.write(s + '\n')
            k += 1


def test_gen_rand_fa_file(n_seq=100, min_len=30, max_len=60, rand_seq_file='random_sequences.fasta'):
    import random

    def generate_random_sequence(length):
        bases = ['A', 'C', 'G', 'T']
        return ''.join(random.choice(bases) for _ in range(length))

    def generate_random_fasta_file(file_path, num_sequences):
        with open(file_path, 'w') as fasta_file:
            for i in range(num_sequences):
                sequence_length = random.randint(min_len, max_len)
                sequence = generate_random_sequence(sequence_length)
                fasta_file.write(f'>seq{i}\n')
                fasta_file.write(sequence + '\n')

    generate_random_fasta_file(rand_seq_file, n_seq)


def test_proc_input():
    input_fasta_file = "./tests/test.fa"
    res_dir = "./test"

    proc_input(input_fasta_file, res_dir, out_bin_file_name = "input.bin.pkl",
               out_boarder_bin_file_name= "input.seqboarder.bin.pkl", debug = True)

    #rand_seq_file = "test_rand_seq.fasta"
    #test_gen_rand_fa_file(rand_seq_file=rand_seq_file)
    #proc_input("test_rand_seq.fasta", out_dir ="..", debug = True)


def test_preproc():
    input_fasta_file = "./tests/test.fa"
    res_dir = "./test"
    config_dict, motif_def_dict = _preproc(input_fasta_file, res_dir)
    print(config_dict)
    print(motif_def_dict)


def test_read_default_config_file():
    config_dict = read_default_config_file(debug=True)
    motid_def_dict = gen_motif_def_dict(config_dict, debug=True)
    print(config_dict)


def test_comp_kmer_hash_taichi() -> Tuple:
    kmer_len = 10
    input_binary_file = "../input.bin.pkl"
    with open(input_binary_file, "rb") as fh:
        seq_np_arr = pickle.load(fh)

    res = comp_kmer_hash_taichi(seq_np_arr, kmer_len)
    uniq_kh_arr, uniq_kh_cnt_arr = count_uniq_hash(res, kmer_len)
    print(res)
    print(uniq_kh_arr, uniq_kh_cnt_arr)


def test_count_uniq_hash():
    seq = "TTTTCGTNCACGACGCTACCTTAAAGCATCCTTCTNTGATACCATAGANNNNNGCAGCTCCTTATCGTTTTAGCTTTCGTATTCGTCTAATCGTCTTTTACTCGACGAAAA"
    kmer_len = 3
    from tests.inimotif import KmerCounter
    kc8 = KmerCounter(kmer_len, revcom_flag=False, unique_kmer_in_seq_mode=False)
    kmer_dict = kc8.scan_seq(seq)
    c8 = Counter(kmer_dict)

    seq_np_arr = dna2arr(seq)
    hash_arr = comp_kmer_hash_taichi(seq_np_arr, kmer_len)
    res = count_uniq_hash(hash_arr, kmer_len)
    c8_new = Counter(dict(zip(res[0], res[1])))

    assert c8 == c8_new

    print(c8)
    print(c8_new)


def test_mask_input():
    seq = "TTTTCGTNCACGACGCTACCTTAAAGCATCCTTCTNTGATACCATAGANNNNNGCAGCTCCTTATCGTTTTAGCTTTCGTATTCGTCTAATCGTCTTTTACTCGACGAAAA"
    kmer_len = 5
    from tests.inimotif import KmerCounter
    kc8 = KmerCounter(kmer_len, revcom_flag=False, unique_kmer_in_seq_mode=False)

    consensus_kh = kc8.kmer2hash(seq[0:5])
    max_hamball_dist = 2

    seq_np_arr = dna2arr(seq)
    seq_np_arr1 = mask_input(seq_np_arr, kmer_len, np.array([consensus_kh]), np.array([max_hamball_dist]))
    print(seq_np_arr1)

    seq_np_arr1[seq_np_arr1==MISSING_VAL] = 4
    alphabet = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
    seq_arr = [alphabet[e] for e in seq_np_arr1]
    print("".join(seq_arr))
    print(seq)


def test_merge_revcom():
    kmer_len = 3
    kh_arr = np.array([0, 2, 10, 11, 17, 18, 19, 23, 27, 33, 36, 38, 41, 43, 46, 51, 53, 57, 59])
    rc_kh_arr = np.array([revcom_hash(i, kmer_len) for i in kh_arr])
    cnt_arr = np.ones_like(kh_arr)

    comm_kh, comm_kh_nat_inds, comm_kh_rc_inds = np.intersect1d(kh_arr, rc_kh_arr, return_indices=True)
    print(f"{comm_kh= }")
    print(f"{comm_kh_nat_inds}")
    print(f"{comm_kh_rc_inds}")
    assert np.all(np.sort(comm_kh_nat_inds) == np.sort(comm_kh_rc_inds))

    cnt_before_merge = sum(cnt_arr)
    merge_kh_arr, merge_kh_cnt_arr = merge_revcom(kh_arr, cnt_arr, kmer_len, keep_lower_hash_flag=True)
    assert all(merge_kh_cnt_arr < 32)
    assert cnt_before_merge == sum(merge_kh_cnt_arr)

    for kh in [10, 17, 36]:
        assert merge_kh_cnt_arr[merge_kh_arr == kh] == 2

    kmer_len = 4
    kh_arr = np.random.choice(256, 1000)
    uniq_kh_arr, uniq_kh_cnt_arr = np.unique(kh_arr, return_counts=True)
    cnt_before_merge = sum(uniq_kh_cnt_arr)
    merge_kh_arr, merge_kh_cnt_arr = merge_revcom(uniq_kh_arr, uniq_kh_cnt_arr, kmer_len, keep_lower_hash_flag=True)
    assert all(merge_kh_cnt_arr < 256/2)
    assert cnt_before_merge == sum(merge_kh_cnt_arr)


def test_kmer2hash():
    kmer = "ACTGA"
    kh = kmer2hash(kmer)

    kc8 = KmerCounter(len(kmer), revcom_flag=False, unique_kmer_in_seq_mode=False)
    assert kh == kc8.kmer2hash(kmer)

    kmer = "ACTACTGGAGGACCTACGTAAGCCACGA"
    kh = kmer2hash(kmer)

    kc8 = KmerCounter(len(kmer), revcom_flag=False, unique_kmer_in_seq_mode=False)
    assert kh == kc8.kmer2hash(kmer)


def test_hash2kmer():
    kmer = "ACTGA"
    kc8 = KmerCounter(len(kmer), revcom_flag=False, unique_kmer_in_seq_mode=False)

    kh = kc8.kmer2hash(kmer)
    assert hash2kmer(kh, len(kmer)) == kmer

    kmer = "ACTACTGGAGGACCTACGTAAGCCACGA"
    kc8 = KmerCounter(len(kmer), revcom_flag=False, unique_kmer_in_seq_mode=False)

    kh = kc8.kmer2hash(kmer)
    assert hash2kmer(kh, len(kmer)) == kmer

def test_mask_ham_ball():
    seqs = "AAAAAAAAAAAAAAAAAAAAAACTAGCTGCCAGTCCCCCCCCCCC"
    seq_np_arr = dna2arr(seqs)[:-1]
    motif_def_dict = init_motif_def_dict("../src/kmap/default_motif_def_table.csv")
    consensus_seq_list = ["AAA", "CCCC"]
    max_ham_dist_list = [0, 0]
    res = mask_ham_ball(seq_np_arr, motif_def_dict, consensus_seq_list, max_ham_dist_list)
    res_str = arr2dna(res)
    assert res_str == "NNNNNNNNNNNNNNNNNNNNNNCTAGCTGCCAGTNNNNNNNNNNN"

    seqs = "AAAAAAAAAAAAAAAAAAAAAACTAGCTGGGGGGGGGGGGGGGGGGGGGGGGGGCCAGTCCCCCCCCCCC"
    seq_np_arr = dna2arr(seqs)[:-1]
    motif_def_dict = init_motif_def_dict("../src/kmap/default_motif_def_table.csv")
    consensus_seq_list = ["AAAAAAA", "CCCCCCCC", "GGGGGGGGG"]
    res = mask_ham_ball(seq_np_arr, motif_def_dict, consensus_seq_list)
    res_str = arr2dna(res)
    assert res_str == "NNNNNNNNNNNNNNNNNNNNNNNTANNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNANNNNNNNNNNNNN"


def test_find_motif():
    input_pkl_file = f"../input.bin.pkl"
    with open(input_pkl_file, "rb") as fh:
        seq_np_arr = pickle.load(fh)
    kmer_len = 8
    motif_def_dict = init_motif_def_dict("../src/kmap/default_motif_def_table.csv")
    max_ham_dist = motif_def_dict[kmer_len].max_ham_dist
    ratio_mu = motif_def_dict[kmer_len].ratio_mu
    ratio_std = motif_def_dict[kmer_len].ratio_std
    ratio_cutoff = motif_def_dict[kmer_len].ratio_cutoff
    p_unif = motif_def_dict[kmer_len].p_uniform
    kmer_cnt_pkl_file = f"./k{kmer_len}.pkl"

    res_dict = find_motif(seq_np_arr, kmer_len, max_ham_dist, p_unif,
                          ratio_mu, ratio_std, ratio_cutoff, rep_mode=True, kmer_cnt_pkl_file=kmer_cnt_pkl_file)
    for kh, values in res_dict.items():
        kh_prop, kh_ratio_log10_pvalue = values
        print(f'{hash2kmer(kh, kmer_len)} {kh_prop*100:02f}% {kh_ratio_log10_pvalue:0.4f}')


def test_sample_disp_kmer():
    consensus_seq_list = ["AACCCTAACCCT", "GGAAGGAA"]
    kmer_len = 12
    motif_def_dict = init_motif_def_dict("../src/kmap/default_motif_def_table.csv")
    out_dir = "../"
    sample_kmer_kh, sample_kmer_count, sample_kmer_label, consensus_list = \
        sample_disp_kmer(consensus_seq_list, kmer_len, motif_def_dict, Path(out_dir))
    #convert hash to kmer
    sample_kmer_list = [hash2kmer(kh, kmer_len) for kh in sample_kmer_kh]
    assert len(sample_kmer_list) == len(sample_kmer_count) == len(sample_kmer_label)
    print(sum(sample_kmer_count))
    print(sum(sample_kmer_count[sample_kmer_label == 0]))
    print(sum(sample_kmer_count[sample_kmer_label == 1]))
    print(sum(sample_kmer_count[sample_kmer_label == 2]))

    print(sample_kmer_label)
    print(sample_kmer_list)

def test_sample_disp_kmer_gen_data(test_dir="./test"):
    def rand_seq(n):
        return random.choices(["A", "C", "G", "T"], k=n)

    def ham_dist(seq1, seq2, n=0):
        # assert len(seq1)==len(seq2)
        if n == 0:
            n = len(seq1)
        dist = 0
        for i in range(n):
            if seq1[i] != seq2[i]:
                dist += 1
        return dist

    def sample_data(conseq1, conseq2, n_seq):
        assert len(conseq1) < len(conseq2)
        label_list = []
        seq_list = []

        conseq = conseq1
        for i in range(n_seq):
            tmpseq = [c for c in conseq] + rand_seq(len(conseq2) - len(conseq1))
            tmp_pos = np.random.randint(0, len(conseq) - 1)
            tmpseq[tmp_pos] = random.sample(["A", "C", "G", "T"], 1)[0]
            seq_list.append("".join(tmpseq))
            label_list.append(0)

        conseq = conseq2
        for i in range(n_seq):
            tmpseq = [c for c in conseq]
            tmp_pos = np.random.randint(0, len(conseq) - 1)
            tmpseq[tmp_pos] = random.sample(["A", "C", "G", "T"], 1)[0]
            seq_list.append("".join(tmpseq))
            label_list.append(1)

        for i in range(2 * n_seq):
            tmpseq = rand_seq(len(conseq2))
            seq_list.append("".join(tmpseq))
            label_list.append(2)

        return seq_list, label_list

    conseq1 = "ACGTGAGGA"
    conseq2 = "CGGACATAGTGA"
    n_seq = 10000
    kmer_len = len(conseq2)
    overwrite_flag = True

    # create test file
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    input_fasta_file = f"{test_dir}/test.fa"
    if not overwrite_flag and os.path.isfile(input_fasta_file):
        seq_list = []
        label_list = []
        with open(input_fasta_file, "r") as fh:
            for rec in SeqIO.parse(fh, "fasta"):
                seq_list.append(str(rec.seq))
                label_list.append(int(rec.name.split("_")[1]))
    else:
        seq_list, label_list = sample_data(conseq1, conseq2, n_seq)
        with open(input_fasta_file, "w") as fh:
            i = 0
            for seq, label in zip(seq_list, label_list):
                fh.write(f">{i}_{label}\n")
                fh.write(f"{seq}\n")
                # print(f"{seq} {label}")
                i = i + 1

    min_k = len(conseq1)
    max_k = len(conseq2) + 1
    user_consensus_list = []
    user_noise_list = []
    main(input_fasta_file, min_k, max_k, user_consensus_list, user_noise_list, out_dir=test_dir, debug=True)

    # todo: collect consensus sequences, merge them

    consensus_seq_list = [conseq1, conseq2]
    motif_def_dict = init_motif_def_dict("../src/kmap/default_motif_def_table.csv")
    out_dir = test_dir
    sample_kmer_kh, sample_kmer_count, sample_kmer_label, consensus_list = \
        sample_disp_kmer(consensus_seq_list, kmer_len, motif_def_dict, Path(out_dir))

    # should be close to n_seq, n_seq, 2*n_seq
    print(f"label0 count: {sum(sample_kmer_count[sample_kmer_label == 0])}")
    print(f"label1 count: {sum(sample_kmer_count[sample_kmer_label == 1])}")
    print(f"label2 count: {sum(sample_kmer_count[sample_kmer_label == 2])}")

    i = 0
    combined_list = sorted(zip(sample_kmer_kh, sample_kmer_count, sample_kmer_label), key=lambda x: x[2])
    for kh, cnt, label in combined_list:
        if label == len(consensus_list):
            conseq = conseq2
        else:
            conseq = consensus_seq_list[label]

        if label == 0:
            h_dist = ham_dist(hash2kmer(kh, kmer_len), conseq, len(conseq1))
        elif label == 1:
            h_dist = ham_dist(hash2kmer(kh, kmer_len), conseq, len(conseq2))
        else:
            h_dist = ham_dist(hash2kmer(kh, kmer_len), conseq, len(conseq2))

        print(f"{i= } {label=} {h_dist= }")
        i += 1

    return sample_kmer_kh, sample_kmer_count, sample_kmer_label, consensus_list


def test_convert_to_block_mat():
    samp_cnts = np.array([2, 1, 4])
    uniq_dist_mat = np.random.randint(0, 100, (3, 3))
    hamdist_mat = _convert_to_block_mat(uniq_dist_mat, samp_cnts)

    print(samp_cnts)
    print(uniq_dist_mat)
    print(hamdist_mat)


def test_cal_samp_kmer_hamdist_mat():
    def rand_seq(n):
        return random.choices(["A", "C", "G", "T"], k=n)
    def ham_dist(seq1, seq2, n=0):
        #assert len(seq1)==len(seq2)
        if n==0:
            n = len(seq1)
        dist = 0
        for i in range(n):
            if seq1[i] != seq2[i]:
                dist += 1
        return dist

    def sample_data(conseq1, conseq2, n_seq):
        assert len(conseq1) < len(conseq2)
        label_list = []
        seq_list = []

        conseq = conseq1
        for i in range(n_seq):
            tmpseq = [c for c in conseq] + rand_seq(len(conseq2) - len(conseq1))
            tmp_pos = np.random.randint(0, len(conseq) - 1)
            tmpseq[tmp_pos] = random.sample(["A", "C", "G", "T"], 1)[0]
            seq_list.append("".join(tmpseq))
            label_list.append(0)

        conseq = conseq2
        for i in range(n_seq):
            tmpseq = [c for c in conseq]
            tmp_pos = np.random.randint(0, len(conseq) - 1)
            tmpseq[tmp_pos] = random.sample(["A", "C", "G", "T"], 1)[0]
            seq_list.append("".join(tmpseq))
            label_list.append(1)

        for i in range(2 * n_seq):
            tmpseq = rand_seq(len(conseq2))
            seq_list.append("".join(tmpseq))
            label_list.append(2)

        return seq_list, label_list

    conseq1 = "ACGTGAGGA"
    conseq2 = "CGGAGAGAGTA"
    n_seq = 50
    kmer_len = len(conseq2)
    overwrite_flag = True

    # create test file
    input_fasta_file = "../test.fa"
    if not overwrite_flag and os.path.isfile(input_fasta_file):
        seq_list = []
        label_list = []
        with open(input_fasta_file, "r") as fh:
            for rec in SeqIO.parse(fh, "fasta"):
                seq_list.append(str(rec.seq))
                label_list.append(int(rec.name.split("_")[1]))
    else:
        seq_list, label_list = sample_data(conseq1, conseq2, n_seq)
        with open(input_fasta_file, "w") as fh:
            i = 0
            for seq, label in zip(seq_list, label_list):
                fh.write(f">{i}_{label}\n")
                fh.write(f"{seq}\n")
                #print(f"{seq} {label}")
                i = i + 1

    min_k = len(conseq1)
    max_k = len(conseq2)+1
    user_consensus_list = []
    user_noise_list = []
    main(input_fasta_file, min_k, max_k, user_consensus_list, user_noise_list, debug=True)

    consensus_seq_list = [conseq1, conseq2]
    motif_def_dict = init_motif_def_dict("../src/kmap/default_motif_def_table.csv")
    out_dir = "../"
    sample_kmer_kh, sample_kmer_count, sample_kmer_label, consensus_list = \
        sample_disp_kmer(consensus_seq_list, kmer_len, motif_def_dict, Path(out_dir))

    # should be close to n_seq, n_seq, 2*n_seq
    print(f"label0 count: {sum(sample_kmer_count[sample_kmer_label==0])}")
    print(f"label1 count: {sum(sample_kmer_count[sample_kmer_label == 1])}")
    print(f"label2 count: {sum(sample_kmer_count[sample_kmer_label == 2])}")

    i = 0
    combined_list = sorted(zip(sample_kmer_kh, sample_kmer_count, sample_kmer_label), key=lambda x: x[2])
    for kh, cnt, label in combined_list:
        if label == len(consensus_list):
            conseq = conseq2
        else:
            conseq = consensus_seq_list[label]

        if label==0:
            h_dist = ham_dist(hash2kmer(kh, kmer_len), conseq, len(conseq1))
        elif label==1:
            h_dist = ham_dist(hash2kmer(kh, kmer_len), conseq, len(conseq2))
        else:
            h_dist = ham_dist(hash2kmer(kh, kmer_len), conseq, len(conseq2))

        print(f"{i= } {label=} {h_dist= }")
        i += 1

    inds = np.argsort(sample_kmer_label)
    sample_kmer_kh = sample_kmer_kh[inds]
    sample_kmer_count = sample_kmer_count[inds]
    sample_kmer_label = sample_kmer_label[inds]

    for kh, label in zip(sample_kmer_kh, sample_kmer_label):
        print(f"label={label} kh={kh} seq={hash2kmer(kh, kmer_len)}")

    uniq_dist_mat = cal_samp_kmer_hamdist_mat(sample_kmer_kh, sample_kmer_count, sample_kmer_label,
                                         consensus_list, kmer_len, uniq_dist_flag=True)
    assert np.all(uniq_dist_mat.diagonal() == 0)
    assert np.all(uniq_dist_mat == np.transpose(uniq_dist_mat))

    for i, conseq in enumerate([conseq1, conseq2]):
        tmpinds = np.where(sample_kmer_label == i)[0]
        tmpinds1 = np.where(sample_kmer_label == 2)[0]
        idx = np.ix_(tmpinds, tmpinds)
        idx1 = np.ix_(tmpinds, tmpinds1)
        print(i)
        assert np.all(uniq_dist_mat[idx] <= motif_def_dict[len(conseq)].max_ham_dist*2)
        assert np.mean(uniq_dist_mat[idx]) < np.mean(uniq_dist_mat[idx1])

    print(uniq_dist_mat)


def test_gen_motif_co_occurence_mat():
    input_fasta_file = "test1.fa"
    conseq_list = ["ACGT", "GAAG"]
    motif_def_dict = init_motif_def_dict("../src/kmap/default_motif_def_table.csv")
    res = gen_motif_co_occurence_mat(input_fasta_file, conseq_list, motif_def_dict, revcom_mode = True)
    assert np.all(res == np.array([[0, 2], [2, 0]]))
    print(res)


def test_knn_smooth():
    def knn_smooth1(dist_mat: np.ndarray, n_neighbor: int) -> np.ndarray:
        """
        perform smooth to the distance matrix by sampling neighbours and take the mean
        Args:
            dist_mat: n x n matrix, each row/column is a point
            n_neighbor: number of neighbors to be sampled for each point
        Returns:
            smoothed distance matrix
        """
        n = len(dist_mat)
        neighbor_inds_mat = np.argpartition(dist_mat, n_neighbor, axis=1)[:, :n_neighbor]

        ret_mat = np.zeros_like(dist_mat, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                tmp_sum = 0.0
                for ii in neighbor_inds_mat[i]:
                    for jj in neighbor_inds_mat[j]:
                        tmp_sum += dist_mat[ii, jj]
                ret_mat[i, j] = tmp_sum / n_neighbor / n_neighbor
                ret_mat[j, i] = ret_mat[i, j]
        return ret_mat

    n_point, n_neighbor = 10, 4
    dist_mat = np.random.randint(low=0, high=100, size=(n_point, n_point))
    dist_mat = np.triu(dist_mat, 1)
    dist_mat += np.transpose(dist_mat)

    new_mat1 = knn_smooth1(dist_mat, n_neighbor)
    new_mat = knn_smooth(dist_mat, n_neighbor)
    print(f"{new_mat= }")
    print(f"{new_mat1= }")
    assert np.all((new_mat - new_mat1) < 1e-4)

def test_merge_consensus_seqs():
    # Test 1
    kmers_example = ["ACGTACGT", "CGTACGT", "TACGTT", "ACGT", "TAC", "CGTA", "ACG","CCTAGGGG", "CTAGGGG", "TAGGGG", "AGG", "GG"]
    res = merge_consensus_seqs(kmers_example)
    assert res == ["CGTACGT", "CTAGGGG"]
    # Test 2


def test_kmap():
    sample_kmer_kh, sample_kmer_count, sample_kmer_label, consensus_list = test_sample_disp_kmer_gen_data()
    kmer_len = 12
    hamdist_mat = cal_samp_kmer_hamdist_mat(sample_kmer_kh, sample_kmer_count, sample_kmer_label, consensus_list, kmer_len, uniq_dist_flag=False)
    ld_data = kmap(hamdist_mat, kmer_len, n_neighbour=20, n_max_iter=100)
    ld_label_arr = _convert_to_block_arr(sample_kmer_label, sample_kmer_count)
    plot_2d_data(ld_data, ld_label_arr, consensus_list)


def test_ex_hamball_kh_arr():
    res_dir = "../test"
    conseq = "CGGAGAGAGTA"
    motif_def_file = "../src/kmap/default_motif_def_table.csv"
    kh_arr, kh_cnt = ex_hamball_kh_arr(res_dir, conseq, motif_def_file=motif_def_file, revcom_mode = True)
    for kh, cnt in zip(kh_arr, kh_cnt):
        print(f"{hash2kmer(kh, len(conseq))}  cnt={cnt}")


def test_ex_hamball():
    res_dir = "./test"
    conseq = "CCATCCATCCATCCA"
    for return_type in ("hash", "kmer", "matrix"):
        output_file = res_dir + "/" + "hamball_" + return_type + ".csv"
        ex_hamball(res_dir, conseq, return_type, output_file)
    draw_logo(res_dir + "/" + "hamball_" + "matrix" + ".csv", res_dir + "/" + "logo.pdf")


def test_scan_motif():
    input_fasta_file = "./tests/test.fa"
    res_dir = "./test"
    debug = False

    _preproc(input_fasta_file, res_dir, debug=debug)
    _scan_motif(res_dir, debug=debug)


def test_visualize_kmers():
    res_dir = "./test"
    debug = True
    _visualize_kmers(res_dir, debug=debug)


if __name__ == "__main__":
    test_proc_input()
    #test_read_default_config_file()
    #test_comp_kmer_hash_taichi()
    #test_count_uniq_hash()
    #test_mask_input()
    #test_merge_revcom()
    #test_hash2kmer()
    #test_kmer2hash()

    #test_find_motif()
    #test_init_motif_def_dict()
    #test_mask_ham_ball()

    #test_main()
    #test_find_motif()
    #test_sample_disp_kmer()

    #test_sample_disp_kmer_gen_data()

    #test_convert_to_block_mat()
    #test_cal_samp_kmer_hamdist_mat()
    #test_gen_motif_co_occurence_mat()
    #test_knn_smooth()

    #test_merge_consensus_seqs()
    #test_kmap()

    #gen_test_fa_file()
    test_preproc()
    #test_ex_hamball_kh_arr()
    #test_ex_hamball()
    #test_scan_motif()
    #test_visualize_kmers()

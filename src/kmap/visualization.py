import bisect

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from .taichi_core import cal_ld_prob_mat_kernel, cal_cross_entropy_kernel, cal_gradient_loss_kernel, knn_smooth_kernel
import taichi as ti

from typing import List
from .kmer_count import FileNameDict
from .motif_discovery import write_lines
from pathlib import Path
import tomllib
import pickle
import click


@click.command(name="visualize_kmers")
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
def visualize_kmers(res_dir: str, debug=False):
    _visualize_kmers(res_dir, debug)


def _visualize_kmers(res_dir: str, debug=False):
    # load config file
    config_file_name = FileNameDict["config_file"]  # config.toml
    config_file_path = Path(res_dir) / config_file_name
    assert config_file_path.exists()
    with open(config_file_path, "rb") as fh:
        config_dict = tomllib.load(fh)

    if not debug:
        debug = config_dict["general"]["debug"]
    n_neighbour = config_dict["visualization"]["n_neighbour"]
    random_seed = config_dict["visualization"]["random_seed"]
    n_max_iter = config_dict["visualization"]["n_max_iter"]
    learning_rate = config_dict["visualization"]["learning_rate"]
    n_best_result = config_dict["visualization"]["n_best_result"]

    if random_seed == "default":
        random_seed = None
    else:
        assert isinstance(random_seed, (int, float))

    sample_kmer_hamdist_mat_file = Path(res_dir) / FileNameDict["sample_kmer_hamdist_mat_file"]
    with open(sample_kmer_hamdist_mat_file, "rb") as fh:
        kmer_len, hamdist_mat, label_arr = pickle.load(fh)

    # dimension reduction
    ld_data = kmap(hamdist_mat, kmer_len, n_neighbour = n_neighbour,
                    n_max_iter = n_max_iter, learning_rate = learning_rate,
                    n_best_result = n_best_result, random_seed = random_seed, debug = debug)
    ld_data_lines = [f"x\ty\tlabel"]
    for x, y, label in zip(ld_data[0], ld_data[1], label_arr):
        ld_data_lines.append(f"{x:3.3f}\t{y:3.3f}\t{int(label)}")
    #for i, label in enumerate(label_arr):
    #    ld_data_lines.append(f"{ld_data[0][i]:3.3f}\t{ld_data[0][1]:3.3f}\t{int(label)}")

    ld_data_file = Path(res_dir) / FileNameDict["ld_data_file"]
    write_lines(ld_data_lines, ld_data_file)
    print("Dimensionality reduction finished. Low dimensional embeddings generated.")

    # plot 2d
    gen_fig_flag = config_dict["visualization"]["gen_fig_flag"]
    if not gen_fig_flag:
        return

    final_conseq_file = Path(res_dir) / FileNameDict["final_conseq_file"]
    assert final_conseq_file.exists()
    with open(final_conseq_file, "r") as fh:
        conseq_list = fh.read().splitlines()

    ld_fig_file_stem = Path(res_dir) / FileNameDict["ld_fig_file_stem"]
    plot_2d_data(ld_data, label_arr, conseq_list,
                       point_size=0.5, point_alpha=0.5, point_color="gray", output_fig_file_stem=str(ld_fig_file_stem))


def knn_smooth(dist_mat: np.ndarray, n_neighbour: int) -> np.ndarray:
    """
    perform smoothing to the distance matrix by sampling neighbours and take the mean
    Args:
        dist_mat: n x n matrix, each row/column is a point
        n_neighbour: number of neighbors to be used for each point
    Returns:
        smoothed distance matrix (dtype=float32)
    """
    n = len(dist_mat)
    neighbor_inds_mat = np.argpartition(dist_mat, n_neighbour, axis=1)[:, :n_neighbour]

    dist_mat = dist_mat.astype("float32")
    ret_mat = np.zeros_like(dist_mat)
    neighbor_inds_mat = neighbor_inds_mat.astype("int32")
    knn_smooth_kernel(dist_mat, neighbor_inds_mat, ret_mat, n, n_neighbour)

    ret_mat += np.transpose(ret_mat)

    return ret_mat

def gradient_loss(hd_prob_mat, ld_prob_mat, ld_data, debug=False):
    ld_data_diff = np.expand_dims(ld_data, 1) - np.expand_dims(ld_data, 0)  # n x n x 2
    prob_mat_diff = hd_prob_mat - ld_prob_mat # n x n
    ld_ratio_mat = ld_prob_mat / (1 - ld_prob_mat) # n x n
    tmp_diff_mat = ld_ratio_mat * prob_mat_diff # n x n

    assert not np.any(np.isnan(tmp_diff_mat))

    n = len(hd_prob_mat)
    ret_mat = np.zeros((n, 2))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ret_mat[i, :] += tmp_diff_mat[i, j] * ld_data_diff[i, j, :]
        if debug:
            print(f"{i= } {ret_mat[i, :]= }")
    return 4.0 * ret_mat


def gradient_loss_taichi(hd_prob_mat, ld_prob_mat, ld_data, debug=False):
    prob_mat_diff = hd_prob_mat - ld_prob_mat # n x n
    ld_ratio_mat = ld_prob_mat / (1 - ld_prob_mat) # n x n
    tmp_diff_mat = ld_ratio_mat * prob_mat_diff # n x n

    assert not np.any(np.isnan(tmp_diff_mat))
    assert tmp_diff_mat.dtype == np.float32
    assert ld_data.dtype == np.float32
    assert len(ld_data) == 2

    n = len(hd_prob_mat)
    ret_mat = np.zeros((2, n), dtype="float32")
    cal_gradient_loss_kernel(tmp_diff_mat, ld_data, ret_mat, n)

    return 4.0 * ret_mat


def cross_entropy(hd_prob_mat, ld_prob_mat) -> float:
    """
    calculate the cross entropy between hd_prob_mat and ld_prob_mat
    Args:
        hd_prob_mat: high dimensional data probability matrix
        ld_prob_mat: low dimensional data probability matrix
    Returns:
        cross entropy, one float number
    """
    tmpmat = - hd_prob_mat * np.log(ld_prob_mat) - (1 - hd_prob_mat) * np.log(1 - ld_prob_mat)
    np.fill_diagonal(tmpmat, 0.0)
    return np.sum(tmpmat)


def cross_entropy_taichi(hd_prob_mat, ld_prob_mat, iter_mat) -> float:
    """
    calculate the cross entropy between hd_prob_mat and ld_prob_mat
    Args:
        hd_prob_mat: high dimensional data probability matrix
        ld_prob_mat: low dimensional data probability matrix
    Returns:
        cross entropy, one float number
    """
    assert hd_prob_mat.dtype == np.float32
    assert ld_prob_mat.dtype == np.float32
    cross_entropy_mat = np.zeros_like(hd_prob_mat)
    i_ind_arr, j_ind_arr = iter_mat[0], iter_mat[1]
    cal_cross_entropy_kernel(hd_prob_mat, ld_prob_mat, len(i_ind_arr), i_ind_arr, j_ind_arr, cross_entropy_mat)
    return np.sum(cross_entropy_mat) * 2


def add_jitter(ld_data: np.ndarray, eps: float) -> np.ndarray:
    """
    add jitters to identical points
    Args:
        ld_data: low dimensional data
        eps: cutoff of identical points
    Returns:
        updated low dimensional data
    """
    def _add_jitter_1d(ld_data, d: int):
        idx = np.argsort(ld_data[:, d])
        sort_arr = ld_data[idx, d]
        tmpinds = np.where(np.diff(sort_arr) < eps)[0]
        ld_data[idx[tmpinds], d] += np.random.normal(0, 0.01, len(tmpinds))
        return ld_data
    ld_data = _add_jitter_1d(ld_data, 0)
    ld_data = _add_jitter_1d(ld_data, 1)
    return ld_data


def sigmoid(dist_mat: np.ndarray, max_val=16.0, change_point=10.0, scale_factor=3.0) -> np.ndarray:
    """
    transform the distance matrix
    Args:
        dist_mat: distance matrix
        max_val: maximum value of the distance
        change_point: change point of the distance distribution
        scale_factor: scale factor for the conversion
    Returns:
        converted distance matrix
    """
    assert max_val > change_point > 0
    assert scale_factor > 0
    return max_val / (1 + np.exp(-scale_factor * (dist_mat - change_point)))


def cal_euclidean_dist2_mat(ld_data: np.ndarray) -> np.ndarray:
    """
    calculate the euclidean distance square for each pair of input data
    Args:
        ld_data: low dimensional data, n x 2 matrix
    Returns:
        Euclidean distance matrix, n x n
    """
    a = ld_data
    b = a.reshape(a.shape[0], 1, a.shape[1])

    return np.einsum('ijk, ijk->ij', a - b, a - b)


def cal_ld_prob_mat(ld_dist2_mat: np.ndarray):
    q = 1.0 / (1 + ld_dist2_mat)
    q = np.minimum(1 - 1e-12, q)
    return q


def cal_ld_prob_mat_taichi(ld_data: np.ndarray, iter_mat: np.ndarray):
    """
    calculate the probability matrix for the low dimensional data
    Args:
        ld_data: 2 x n matrix
        iter_mat: iteration index (2 x n matrix) for i,j pairs such that i<j, iter_mat[0] are index for i, iter_mat[1] are index for j
    Returns:
        n x n probability matrix
    """
    assert ld_data.shape[0] == 2
    assert ld_data.dtype == np.float32
    assert iter_mat.dtype == np.int32
    n = len(ld_data[0])

    x_arr, y_arr = ld_data[0], ld_data[1]
    i_ind_arr, j_ind_arr = iter_mat[0], iter_mat[1]
    prob_mat = np.ones((n, n), dtype="float32")
    cal_ld_prob_mat_kernel(x_arr, y_arr, len(i_ind_arr), i_ind_arr, j_ind_arr, prob_mat)
    eps = 1e-3
    prob_mat = np.minimum(prob_mat, 1 - eps)
    prob_mat = np.maximum(prob_mat, eps)
    return prob_mat


def kmap(hamdist_mat: np.ndarray, kmer_len: int, n_neighbour=20,
         n_max_iter=2500, learning_rate=0.01, n_best_result=10, random_seed=None, debug=True) -> np.ndarray:
    trans_dist_mat = knn_smooth(dist_mat=hamdist_mat, n_neighbour=n_neighbour)
    trans_dist_mat = sigmoid(trans_dist_mat, 16.0, change_point=kmer_len/2, scale_factor=0.2 * kmer_len - 0.2)
    print("distance smoothing finished.")
    ld_data = umap(trans_dist_mat, n_max_iter=n_max_iter, learning_rate=learning_rate,
                   n_best_result=n_best_result, random_seed=random_seed, debug=debug)
    print("optimization finished.")
    return  ld_data


def umap(hd_dist_mat: np.ndarray, n_max_iter=2500, learning_rate=0.01, n_best_result=10, random_seed=None, debug=True) -> np.ndarray:
    """
    perform dimension reduction for kmer distance matrix
    Args:
        hd_dist_mat: distance matrix of kmers (high dimensional data)
        n_max_iter: maximum number of iterations in the optimization
        learning_rate: learning rate in the optimization
        n_best_result: record the 10 best results
        random_seed: random seed for the optimization, None means random seed
    Returns: low dimensional representation of kmers in the form of n x 2 matrix
    """
    np.random.seed(random_seed)

    n_data = len(hd_dist_mat)
    sigma0 = 0.5

    iter_mat = np.array([(i, j) for i in range(n_data) for j in range(n_data) if i < j]).T.astype("int32")
    iter_mat = np.ascontiguousarray(iter_mat)

    hd_prob_mat = np.exp(-hd_dist_mat / sigma0).astype("float32")

    # init low dimensional representation
    ld_data = np.random.randn(2, n_data).astype("float32")
    best_res_list = [(np.array([np.inf]), np.random.randn(2, n_data).astype("float32")) for _ in range(n_best_result)]

    loss = np.inf
    for i_iter in range(n_max_iter):
        if debug:
            print(f"{i_iter= } {loss= }")

        ld_prob_mat = cal_ld_prob_mat_taichi(ld_data, iter_mat)
        curr_loss = cross_entropy_taichi(hd_prob_mat, ld_prob_mat, iter_mat)

        if curr_loss < best_res_list[-1][0]:
            tmpres = best_res_list[-1]
            tmpres[0][0] = curr_loss
            tmpres[1][:] = ld_data[:]
            best_res_list = best_res_list[:-1]
            bisect.insort_right(best_res_list, tmpres, key=lambda x: x[0][0])

        if abs(loss - curr_loss) < 1e-7 * abs(curr_loss):
            break

        loss = curr_loss
        grad_loss = gradient_loss_taichi(hd_prob_mat, ld_prob_mat, ld_data)  # O(n^2)

        ld_data += (-grad_loss * learning_rate)
        ld_data = add_jitter(ld_data, eps=0.1) # nlog(n)

    if debug:
        print()
        print("Best results")
        for i in range(n_best_result):
            print(f"{i= } loss={best_res_list[i][0][0]:0.2f}")

    ld_data = best_res_list[0][1]
    return ld_data


def plot_2d_data(ld_data: np.ndarray, label_arr: np.ndarray=None, conseq_list: List[str]=[], cmap: str = "Dark2",
                 point_size=0.5, point_alpha=0.5, point_color="gray", output_fig_file_stem: str|Path = None):
    # cmap: 'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
    #                       'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
    #                       'tab20c'
    assert ld_data.shape[0] == 2
    assert len(ld_data[0]) == len(label_arr)
    x_arr, y_arr = ld_data[0], ld_data[1]

    if cmap=="Dark2":
        cmap = ListedColormap(plt.get_cmap("Dark2").colors[:7]) # remove the last gray color

    # create the figure
    fig, ax = plt.subplots(figsize=(15, 15))  # 15 inch

    if label_arr is None:
        plt.scatter(x_arr, y_arr, s=point_size, c=point_color)
        if output_fig_file_stem:
            plt.savefig(output_fig_file_stem + ".png", format="png")
            plt.savefig(output_fig_file_stem + ".pdf", format="pdf")
        plt.show()
        return

    max_label = max(label_arr)
    random_kmer_inds = label_arr == max_label
    motif_kmer_inds = np.logical_not(random_kmer_inds)

    if len(conseq_list) == 0:
        conseq_list = [f"motif-{i}" for i in range(max_label)]
    else:
        assert len(conseq_list) == max_label
        conseq_list = [f"m{i}-{conseq_list[i]}" for i in range(max_label)]

    ax.scatter(x_arr[random_kmer_inds], y_arr[random_kmer_inds],
                s=point_size, c=point_color, alpha=point_alpha)
    scatter = ax.scatter(x_arr[motif_kmer_inds], y_arr[motif_kmer_inds],
                s=point_size*1.2, c=label_arr[motif_kmer_inds], alpha=0.9, cmap=cmap)

    handles, labels = scatter.legend_elements()
    ax.legend(handles, conseq_list, loc="upper right", title="motifs")

    if output_fig_file_stem:
        plt.savefig(output_fig_file_stem + ".png", format="png")
        plt.savefig(output_fig_file_stem + ".pdf", format="pdf")

    plt.show()



if __name__=="__main__":

    GPU_MODE = False
    ti.init(arch=ti.cuda, default_ip=ti.i64)
    if ti.cfg.arch == ti.cuda:
        GPU_MODE = True
        print("GPU is available")
    else:
        GPU_MODE = False
        print("GPU is not available")

    hd_dim = 3
    hd_sigma = 50
    hd_data_0 = np.random.normal(0.0, 10, (100, hd_dim)) + np.random.normal(0, hd_sigma, hd_dim)
    hd_data_1 = np.random.normal(0.0, 10, (100, hd_dim)) + np.random.normal(0, hd_sigma, hd_dim)
    hd_data_2 = np.random.uniform(-2.0 * hd_sigma, 2.0 * hd_sigma, (100, hd_dim))
    hd_data = np.vstack((hd_data_0, hd_data_1, hd_data_2))
    print(hd_data.shape)

    hd_dist_mat = np.sqrt(cal_euclidean_dist2_mat(hd_data))
    print(hd_dist_mat.shape)

    #plt.scatter(hd_data[:, 0], hd_data[:, 1])
    #plt.show()

    ld_data = kmap(hd_dist_mat, 2*hd_sigma, 3 * 10, n_max_iter=50).T
    inds = np.arange(200)
    plt.scatter(ld_data[inds, 0], ld_data[inds, 1], color="r")
    inds = np.arange(200, 300)
    plt.scatter(ld_data[inds, 0], ld_data[inds, 1], color="b")
    plt.show()











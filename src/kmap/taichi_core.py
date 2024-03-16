import taichi as ti

@ti.func
def kmer2hash_taichi_uint32(arr: ti.types.ndarray(dtype=ti.u8), arr_size: int, st_pos: int, k: int,
                            hash_arr: ti.types.ndarray(dtype=ti.u32),
                            invalid_hash: ti.types.u32,
                            missing_val: ti.types.u32):
    # hash_arr[st_pos] = invalid_hash
    invalid_hash_flag = 0
    if st_pos + k > arr_size:
        invalid_hash_flag = 1

    kh = ti.u32(0)
    for i in range(k):
        if arr[st_pos + i] == missing_val:
            invalid_hash_flag = 1
        kh = kh << 2
        kh += arr[st_pos + i]
    hash_arr[st_pos] = kh

    if invalid_hash_flag > 0:
        hash_arr[st_pos] = invalid_hash


@ti.kernel
def kmer2hash_kernel_uint32(arr: ti.types.ndarray(dtype=ti.u8), arr_size: int, k: int,
                            hash_arr: ti.types.ndarray(dtype=ti.u32),
                            invalid_hash_arr: ti.types.ndarray(dtype=ti.u32),
                            missing_val_arr: ti.types.ndarray(dtype=ti.u8)):
    for i in range(arr_size):
        kmer2hash_taichi_uint32(arr, arr_size, i, k, hash_arr, invalid_hash_arr[0], missing_val_arr[0])


@ti.func
def kmer2hash_taichi_uint64(arr: ti.types.ndarray(dtype=ti.u8), arr_size: int, st_pos: int, k: int,
                            hash_arr: ti.types.ndarray(dtype=ti.u64), invalid_hash: ti.u64,
                            missing_val: ti.u8):
    # hash_arr[st_pos] = invalid_hash
    invalid_hash_flag = 0
    if st_pos + k > arr_size:
        invalid_hash_flag = 1

    kh = ti.u64(0)
    for i in range(k):
        if arr[st_pos + i] == missing_val:
            invalid_hash_flag = 1
        kh = kh << 2
        kh += arr[st_pos + i]
    hash_arr[st_pos] = kh

    if invalid_hash_flag > 0:
        hash_arr[st_pos] = invalid_hash


@ti.kernel
def kmer2hash_kernel_uint64(arr: ti.types.ndarray(dtype=ti.u8), arr_size: int, k: int,
                            hash_arr: ti.types.ndarray(dtype=ti.u64),
                            invalid_hash_arr: ti.types.ndarray(dtype=ti.u64),
                            missing_val_arr: ti.types.ndarray(dtype=ti.u8)):
    for i in range(arr_size):
        kmer2hash_taichi_uint64(arr, arr_size, i, k, hash_arr, invalid_hash_arr[0], missing_val_arr[0])

@ti.func
def cal_ham_dist_uint32(hash1: ti.u32, hash2: ti.u32, kmer_len: int):
    xor_result = hash1 ^ hash2
    twobit_mask = ti.cast(3, ti.u32)
    hamming_dist = 0
    for _ in range(kmer_len):
        cmp_res = xor_result & twobit_mask
        hamming_dist += cmp_res != 0
        xor_result >>= 2
    return hamming_dist


@ti.kernel
def cal_ham_dist_kernel_uint32(hash_arr: ti.types.ndarray(dtype=ti.u32),
                               target_hash: ti.types.ndarray(dtype=ti.u32),
                               ham_dist_arr: ti.types.ndarray(dtype=ti.u8),
                               hash_arr_size: int,
                               kmer_len: int):
    for i in range(hash_arr_size):
        ham_dist_arr[i] = ti.cast(cal_ham_dist_uint32(hash_arr[i], target_hash[0], kmer_len), ti.u8)


@ti.func
def cal_ham_dist_uint64(hash1: ti.u64, hash2: ti.u64, kmer_len: int):
    xor_result = hash1 ^ hash2
    twobit_mask = ti.cast(3, ti.u64)
    hamming_dist = 0
    for _ in range(kmer_len):
        cmp_res = xor_result & twobit_mask
        hamming_dist += cmp_res != 0
        xor_result >>= 2
    return hamming_dist


@ti.kernel
def cal_ham_dist_kernel_uint64(hash_arr: ti.types.ndarray(dtype=ti.u64),
                               target_hash: ti.types.ndarray(dtype=ti.u64),
                               ham_dist_arr: ti.types.ndarray(dtype=ti.u8),
                               hash_arr_size: int,
                               kmer_len: int):
    for i in range(hash_arr_size):
        ham_dist_arr[i] = ti.cast(cal_ham_dist_uint64(hash_arr[i], target_hash[0], kmer_len), ti.u8)


# newly added on 11.10.2023, not tested yet, -------------- begin ----------------------------
@ti.func
def cal_partial_ham_dist_head_uint32(kmer_hash: ti.u32, conseq_kh: ti.u32, kmer_len: int, conseq_len: int):
    # match kmer_hash with conseq_kh from the head, kmer_len is greater than conseq_len
    for _ in range(kmer_len - conseq_len):
        kmer_hash >>= 2
    return cal_ham_dist_uint32(kmer_hash, conseq_kh, conseq_len)


@ti.kernel
def cal_partial_ham_dist_head_kernel_uint32(hash_arr: ti.types.ndarray(dtype=ti.u32),
                               target_hash: ti.types.ndarray(dtype=ti.u32),
                               ham_dist_arr: ti.types.ndarray(dtype=ti.u8),
                               hash_arr_size: int,
                               kmer_len: int,
                               conseq_len: int):
    for i in range(hash_arr_size):
        ham_dist_arr[i] = ti.cast(cal_partial_ham_dist_head_uint32(hash_arr[i], target_hash[0], kmer_len, conseq_len), ti.u8)


@ti.func
def cal_partial_ham_dist_tail_uint32(kmer_hash: ti.u32, conseq_kh: ti.u32, kmer_len: int, conseq_len: int):
    # match kmer_hash with conseq_kh from the head, kmer_len is greater than conseq_len
    return cal_ham_dist_uint32(kmer_hash, conseq_kh, conseq_len)


@ti.kernel
def cal_partial_ham_dist_tail_kernel_uint32(hash_arr: ti.types.ndarray(dtype=ti.u32),
                               target_hash: ti.types.ndarray(dtype=ti.u32),
                               ham_dist_arr: ti.types.ndarray(dtype=ti.u8),
                               hash_arr_size: int,
                               kmer_len: int,
                               conseq_len: int):
    for i in range(hash_arr_size):
        ham_dist_arr[i] = ti.cast(cal_partial_ham_dist_tail_uint32(hash_arr[i], target_hash[0], kmer_len, conseq_len), ti.u8)


@ti.func
def cal_partial_ham_dist_head_uint64(kmer_hash: ti.u64, conseq_kh: ti.u64, kmer_len: int, conseq_len: int):
    # match kmer_hash with conseq_kh from the head, kmer_len is greater than conseq_len
    for _ in range(kmer_len - conseq_len):
        kmer_hash >>= 2
    return cal_ham_dist_uint64(kmer_hash, conseq_kh, conseq_len)


@ti.kernel
def cal_partial_ham_dist_head_kernel_uint64(hash_arr: ti.types.ndarray(dtype=ti.u64),
                               target_hash: ti.types.ndarray(dtype=ti.u64),
                               ham_dist_arr: ti.types.ndarray(dtype=ti.u8),
                               hash_arr_size: int,
                               kmer_len: int,
                               conseq_len: int):
    for i in range(hash_arr_size):
        ham_dist_arr[i] = ti.cast(cal_partial_ham_dist_head_uint64(hash_arr[i], target_hash[0], kmer_len, conseq_len), ti.u8)


@ti.func
def cal_partial_ham_dist_tail_uint64(kmer_hash: ti.u64, conseq_kh: ti.u64, kmer_len: int, conseq_len: int):
    # match kmer_hash with conseq_kh from the head, kmer_len is greater than conseq_len
    return cal_ham_dist_uint64(kmer_hash, conseq_kh, conseq_len)


@ti.kernel
def cal_partial_ham_dist_tail_kernel_uint64(hash_arr: ti.types.ndarray(dtype=ti.u64),
                               target_hash: ti.types.ndarray(dtype=ti.u64),
                               ham_dist_arr: ti.types.ndarray(dtype=ti.u8),
                               hash_arr_size: int,
                               kmer_len: int,
                               conseq_len: int):
    for i in range(hash_arr_size):
        ham_dist_arr[i] = ti.cast(cal_partial_ham_dist_tail_uint64(hash_arr[i], target_hash[0], kmer_len, conseq_len), ti.u8)

# newly added on 11.10.2023, not tested yet, -------------- end ----------------------------

@ti.func
def revcom_hash_uint32(in_hash: ti.u32,
                       mask: ti.u32,
                       twobit_mask: ti.u32,
                       k: int):
    com_hash = mask - in_hash  # complement hash
    ret_hash = twobit_mask & com_hash
    for i in range(k - 1):
        ret_hash = ret_hash << 2
        com_hash = com_hash >> 2
        ret_hash += twobit_mask & com_hash
    return ret_hash


@ti.func
def revcom_hash_uint64(in_hash: ti.u64,
                       mask: ti.u64,
                       twobit_mask: ti.u64,
                       k: int):
    com_hash = mask - in_hash  # complement hash
    ret_hash = twobit_mask & com_hash
    for i in range(k - 1):
        ret_hash = ret_hash << 2
        com_hash = com_hash >> 2
        ret_hash += twobit_mask & com_hash
    return ret_hash


@ti.kernel
def revcom_hash_kernel_uint32(in_hash_arr: ti.types.ndarray(dtype=ti.u32),
                              out_hash_arr: ti.types.ndarray(dtype=ti.u32),
                              mask_arr: ti.types.ndarray(dtype=ti.u32),
                              kmer_len: int, in_hash_arr_size: int):
    for i in range(in_hash_arr_size):
        out_hash_arr[i] = revcom_hash_uint32(in_hash_arr[i], mask_arr[0], mask_arr[1], kmer_len)


@ti.kernel
def revcom_hash_kernel_uint64(in_hash_arr: ti.types.ndarray(dtype=ti.u64),
                              out_hash_arr: ti.types.ndarray(dtype=ti.u64),
                              mask_arr: ti.types.ndarray(dtype=ti.u64),
                              kmer_len: int, in_hash_arr_size: int):
    for i in range(in_hash_arr_size):
        out_hash_arr[i] = revcom_hash_uint64(in_hash_arr[i], mask_arr[0], mask_arr[1], kmer_len)


@ti.func
def cal_avg_dist(dist_mat: ti.types.ndarray(dtype=ti.f32),
                 neighbor_inds_mat: ti.types.ndarray(dtype=ti.i32),
                 new_dist_mat: ti.types.ndarray(dtype=ti.f32),
                 i: int, j: int, n_neighbour: int):
    tmp_sum = 0.0
    for ii in range(n_neighbour):
        for jj in range(n_neighbour):
            tmp_sum += dist_mat[neighbor_inds_mat[i, ii], neighbor_inds_mat[j, jj]]
    new_dist_mat[i, j] = tmp_sum / n_neighbour / n_neighbour


@ti.kernel
def knn_smooth_kernel(dist_mat: ti.types.ndarray(dtype=ti.f32),
                      neighbor_inds_mat: ti.types.ndarray(dtype=ti.i32),
                      new_dist_mat: ti.types.ndarray(dtype=ti.f32),
                      n_point: int, n_neighbour: int):
    for k in range(n_point * n_point):
        i = k // n_point
        j = k % n_point
        if i >= j:
            continue
        cal_avg_dist(dist_mat, neighbor_inds_mat, new_dist_mat, i, j, n_neighbour)


@ti.func
def cal_ld_prob_pair(xi: ti.f32, xj: ti.f32, yi: ti.f32, yj: ti.f32):
    dist2 = (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj)
    dist2 = 1 / (1 + dist2)
    return dist2


@ti.kernel
def cal_ld_prob_mat_kernel(x_arr: ti.types.ndarray(dtype=ti.f32),
                           y_arr: ti.types.ndarray(dtype=ti.f32),
                           n_iter: int,
                           i_ind_arr: ti.types.ndarray(dtype=ti.i32),
                           j_ind_arr: ti.types.ndarray(dtype=ti.i32),
                           prob_mat: ti.types.ndarray(dtype=ti.f32)):
    for ind in range(n_iter):
        i = i_ind_arr[ind]
        j = j_ind_arr[ind]
        prob_mat[i, j] = cal_ld_prob_pair(x_arr[i], x_arr[j], y_arr[i], y_arr[j])
        prob_mat[j, i] = prob_mat[i, j]

@ti.kernel
def cal_cross_entropy_kernel(hd_prob_mat: ti.types.ndarray(dtype=ti.f32),
                             ld_prob_mat: ti.types.ndarray(dtype=ti.f32),
                             n_iter: int,
                             i_ind_arr: ti.types.ndarray(dtype=ti.i32),
                             j_ind_arr: ti.types.ndarray(dtype=ti.i32),
                             cross_entropy_mat: ti.types.ndarray(dtype=ti.f32)):
    eps = 1e-10
    ld_prob = 0.0
    for ind in range(n_iter):
        i, j = i_ind_arr[ind], j_ind_arr[ind]
        if ld_prob_mat[i, j] < eps:
            ld_prob = eps
        elif ld_prob_mat[i, j] > 1 - eps:
            ld_prob = 1 - eps
        else:
            ld_prob = ld_prob_mat[i, j]

        ret_val = 0.0
        if hd_prob_mat[i, j] < eps:
            ret_val = - ti.log(1-ld_prob)
        elif hd_prob_mat[i, j] > 1 - eps:
            ret_val = - ti.log(ld_prob)
        else:
            ret_val = cal_cross_entropy(hd_prob_mat[i, j], ld_prob)

        cross_entropy_mat[i, j] = ret_val


@ti.func
def cal_cross_entropy(hd_prob: ti.f32, ld_prob: ti.f32):
    return - hd_prob * ti.log(ld_prob) - (1 - hd_prob) * ti.log(1 - ld_prob)

@ti.kernel
def cal_gradient_loss_kernel(tmp_diff_mat: ti.types.ndarray(dtype=ti.f32), # n x n
                             ld_data: ti.types.ndarray(dtype=ti.f32), # 2 x n
                             ret_mat: ti.types.ndarray(dtype=ti.f32), # 2 x n
                             n: int):
    for i in range(n):
        ret_mat[0, i] = cal_gradient_loss(tmp_diff_mat, ld_data, 0, n, i)
        ret_mat[1, i] = cal_gradient_loss(tmp_diff_mat, ld_data, 1, n, i)

@ti.func
def cal_gradient_loss(diff_mat: ti.types.ndarray(dtype=ti.f32),
                      ld_data: ti.types.ndarray(dtype=ti.f32),
                      k: int, # kth dimension
                      n: int,
                      i: int):
    ret_val = 0.0
    for j in range(n):
        if i == j:
            continue
        else:
            ret_val += diff_mat[i, j] * (ld_data[k, i] - ld_data[k, j])
    return ret_val


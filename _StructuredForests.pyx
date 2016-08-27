__author__ = 'artanis'

import math
import numpy as N

cimport numpy as N


ctypedef N.int32_t C_INT32
#ctypedef N.float64_t C_FLOAT64
ctypedef N.float32_t C_FLOAT32


def build_feature_table(shrink, p_size, n_cell, n_ch):
    p_size /= shrink

    reg_tb = []
    for i in xrange(p_size):
        for j in xrange(p_size):
            for k in xrange(n_ch):
                reg_tb.append([i, j, k])

    half_cell_size = int(round(p_size / (2.0 * n_cell)))
    grid_pos = [int(round((i + 1) * (p_size + 2 * half_cell_size - 1) / \
                          (n_cell + 1.0) - half_cell_size))
                for i in xrange(n_cell)]
    grid_pos = [(r, c) for r in grid_pos for c in grid_pos]

    ss_tb = []
    for i in xrange(n_cell ** 2):
        for j in xrange(i + 1, n_cell ** 2):
            for z in xrange(n_ch):
                x1, y1 = grid_pos[i]
                x2, y2 = grid_pos[j]
                ss_tb.append([x1, y1, x2, y2, z])

    return N.asarray(reg_tb, dtype=N.int32), \
           N.asarray(ss_tb, dtype=N.int32)


def find_leaves(float[:, :, :] src, float[:, :, :] reg_ch,
                float[:, :, :] ss_ch,
                int shrink, int p_size, int g_size, int n_cell, int stride,
                int n_tree_eval,
                float[:, :] thrs, int[:, :] fids, int[:, :] cids):
    cdef int n_ftr_ch = reg_ch.shape[2]
    cdef int height = src.shape[0] - p_size, width = src.shape[1] - p_size
    cdef int n_tree = cids.shape[0], n_node_per_tree = cids.shape[1]
    cdef int n_reg_dim = (p_size / shrink) ** 2 * n_ftr_ch
    cdef int i, j, k, x1, x2, y1, y2, z, tree_idx, node_idx, ftr_idx
    cdef float ftr
    cdef int[:, :] reg_tb, ss_tb
    cdef N.ndarray[C_INT32, ndim=3] lids_arr

    reg_tb, ss_tb = build_feature_table(shrink, p_size, n_cell, n_ftr_ch)

    lids_arr = N.zeros((src.shape[0], src.shape[1], n_tree_eval), dtype=N.int32)
    cdef int[:, :, :] lids = lids_arr

    with nogil:
        for i from 0 <= i < height by stride:
            for j from 0 <= j < width by stride:
                for k from 0 <= k < n_tree_eval:
                    tree_idx = ((i + j) / stride % 2 * n_tree_eval + k) % n_tree
                    node_idx = 0

                    while cids[tree_idx, node_idx] != 0:
                        ftr_idx = fids[tree_idx, node_idx]

                        if ftr_idx >= n_reg_dim:
                            x1 = ss_tb[ftr_idx - n_reg_dim, 0] + i / shrink
                            y1 = ss_tb[ftr_idx - n_reg_dim, 1] + j / shrink
                            x2 = ss_tb[ftr_idx - n_reg_dim, 2] + i / shrink
                            y2 = ss_tb[ftr_idx - n_reg_dim, 3] + j / shrink
                            z = ss_tb[ftr_idx - n_reg_dim, 4]

                            ftr = ss_ch[x1, y1, z] - ss_ch[x2, y2, z]
                        else:
                            x1 = reg_tb[ftr_idx, 0] + i / shrink
                            y1 = reg_tb[ftr_idx, 1] + j / shrink
                            z = reg_tb[ftr_idx, 2]

                            ftr = reg_ch[x1, y1, z]

                        if ftr < thrs[tree_idx, node_idx]:
                            node_idx = cids[tree_idx, node_idx] - 1
                        else:
                            node_idx = cids[tree_idx, node_idx]

                    lids[i, j, k] = tree_idx * n_node_per_tree + node_idx

    return lids_arr


def build_neigh_table(g_size):
    dxy = N.c_[
        [1, 1, -1, -1], 
        [1, -1, 1, -1],
    ].astype('i4')
    idx = N.mgrid[:g_size, :g_size]
    idx = idx.transpose(1,2,0).reshape(g_size,g_size,1,2) + dxy
    idx[idx<0] = 0
    idx[idx>=g_size] = g_size-1
    return idx



def predict_sharpen(float[:, :, :] src, int[:, :, :] lids, int sharpen,
            int p_size, int g_size, int stride,  int n_tree_eval, int n_bnd,
            int[:] n_seg, int[:, :, :] segs, int[:] edge_bnds, int[:] edge_pts):
    cdef int height = src.shape[0] - p_size, width = src.shape[1] - p_size
    cdef int depth = src.shape[2], border = (p_size - g_size) / 2
    cdef int n_s, max_n_s = N.max(n_seg)
    cdef int i, j, k, m, n, p, begin, end
    cdef int leaf_idx, x1, x2, y1, y2, best_seg
    cdef float err, min_err
    cdef N.ndarray[C_FLOAT32, ndim=2] dst_arr
    cdef N.ndarray[C_INT32, ndim=5]   segs_arr

    segs_arr = N.zeros((height, width, n_tree_eval, g_size, g_size), dtype=N.int32)
    cdef int[:,:,:,:,:] _segs = segs_arr
    cdef int[:,:] patch = N.zeros((g_size, g_size), dtype=N.int32)
    cdef float[:] count = N.zeros((max_n_s,), dtype=N.float32),
    cdef float[:, :] mean = N.zeros((max_n_s, depth), dtype=N.float32)
    cdef int[:, :, :, :] neigh_tb = build_neigh_table(g_size)

    dst_arr = N.zeros((src.shape[0], src.shape[1]), dtype=N.float32)
    cdef float[:, :] dst = dst_arr

    with nogil:
        for i from 0 <= i < height by stride:
            for j from 0 <= j < width by stride:
                for k from 0 <= k < n_tree_eval:
                    leaf_idx = lids[i, j, k]

                    begin = edge_bnds[leaf_idx * n_bnd]
                    end = edge_bnds[leaf_idx * n_bnd + sharpen + 1]
                    if begin == end:
                        continue

                    n_s = n_seg[leaf_idx]
                    if n_s == 1:
                        continue

                    patch[:, :] = segs[leaf_idx]
                    count[:] = 0.0
                    mean[:] = 0.0

                    # compute color model for each segment using every other pixel
                    for m from 0 <= m < g_size by 2:
                        for n from 0 <= n < g_size by 2:
                            count[patch[m, n]] += 1.0

                            for p from 0 <= p < depth:
                                mean[patch[m, n], p] += \
                                    src[i + m + border, j + n + border, p]

                    for m from 0 <= m < n_s:
                        for n from 0 <= n < depth:
                            mean[m, n] /= count[m]

                    # update segment according to local color values
                    end = edge_bnds[leaf_idx * n_bnd + sharpen]
                    for m from begin <= m < end:
                        min_err = 1e10
                        best_seg = -1

                        x1 = edge_pts[m] / g_size
                        y1 = edge_pts[m] % g_size

                        for n from 0 <= n < 4:
                            x2 = neigh_tb[x1, y1, n, 0]
                            y2 = neigh_tb[x1, y1, n, 1]

                            if patch[x2, y2] == best_seg:
                                continue

                            err = 0.0
                            for p from 0 <= p < depth:
                                err += (src[x1 + i + border, y1 + j + border, p] -
                                        mean[patch[x2, y2], p]) ** 2

                            if err < min_err:
                                min_err = err
                                best_seg = patch[x2, y2]

                        patch[x1, y1] = best_seg

                    # convert mask to edge maps (examining expanded set of pixels)
                    end = edge_bnds[leaf_idx * n_bnd + sharpen + 1]
                    for m from begin <= m < end:
                        x1 = edge_pts[m] / g_size
                        y1 = edge_pts[m] % g_size

                        for n from 0 <= n < 4:
                            x2 = neigh_tb[x1, y1, n, 0]
                            y2 = neigh_tb[x1, y1, n, 1]

                            if patch[x1, y1] != patch[x2, y2]:
                                dst[x1 + i, y1 + j] += 1.0
                                break
                    _segs[i,j,k,:,:] = patch

    return dst_arr, segs_arr



def predict_no_sharpen(N.ndarray[C_FLOAT32, ndim=3] src,
                 N.ndarray[C_INT32, ndim=3]   lids,
                 int p_size, int g_size, int stride, int n_tree_eval, int n_bnd,
                 N.ndarray[C_INT32, ndim=1] edge_bnds,
                 N.ndarray[C_INT32, ndim=1] edge_pts):
    cdef int i, j, k, m, begin, end
    cdef int leaf_idx, loc, x1, y1
    cdef N.ndarray[C_FLOAT32, ndim=2] dst

    dst = N.zeros((src.shape[0], src.shape[1]), dtype=N.float32)

    for i in xrange(0, src.shape[0] - p_size, stride):
        for j in xrange(0, src.shape[1] - p_size, stride):
            for k in xrange(n_tree_eval):
                leaf_idx = lids[i, j, k]

                begin = edge_bnds[leaf_idx * n_bnd]
                end = edge_bnds[leaf_idx * n_bnd + 1]
                if begin == end:
                    continue

                for m in xrange(begin, end):
                    loc = edge_pts[m]
                    x1 = loc / g_size + i
                    y1 = loc % g_size + j
                    dst[x1, y1] += 1.0
    return dst






cdef inline float bilinear_interp(float[:, :] img, float x, float y) nogil:
    """
    Return img[y, x] via bilinear interpolation
    """

    cdef int h = img.shape[0], w = img.shape[1]

    x = min(max(x,0), w-1.001)
    y = min(max(y,0), h-1.001)

    cdef int x0 = int(x), y0 = int(y), x1 = x0 + 1, y1 = y0 + 1
    cdef float dx0 = x - x0, dy0 = y - y0, dx1 = 1 - dx0, dy1 = 1 - dy0

    return img[y0, x0] * dx1 * dy1 + img[y0, x1] * dx0 * dy1 + \
           img[y1, x0] * dx1 * dy0 + img[y1, x1] * dx0 * dy0


def non_maximum_supr(float[:, :] E0, float[:, :] O, int r, int s, float m):
    """
    Non-Maximum Suppression

    :param E0: original edge map
    :param O: orientation map
    :param r: radius for nms suppression
    :param s: radius for suppress boundaries
    :param m: multiplier for conservative suppression
    :return: suppressed edge map
    """

    cdef int h = E0.shape[0], w = E0.shape[1], x, y, d
    cdef float e, e0, co, si
    cdef N.ndarray[C_FLOAT32, ndim=2] E_arr = N.zeros((h, w), dtype=N.float32)
    cdef float[:, :] E = E_arr
    cdef float[:, :] C = N.cos(O), S = N.sin(O)

    with nogil:
        # suppress edges where edge is stronger in orthogonal direction
        for y from 0 <= y < h:
            for x from 0 <= x < w:
                e = E[y, x] = E0[y, x]
                if e == 0:
                    continue

                e *= m
                co = C[y, x]
                si = S[y, x]

                for d from -r <= d <= r:
                    if d == 0:  continue     
                    e0 = bilinear_interp(E0, x + d * co, y + d * si)
                    if e >= e0: continue     
                    E[y, x] = 0             # suppress
                    break

        # suppress noisy edge estimates near boundaries of image
        s = w / 2 if s > w / 2 else s
        s = h / 2 if s > h / 2 else s

        for x from 0 <= x < s:
            for y from 0 <= y < h:
                E[y, x] *= x / <float>s
                E[y, w - 1 - x] *= x / <float>s

        for x from 0 <= x < w:
            for y from 0 <= y < s:
                E[y, x] *= y / <float>s
                E[h - 1 - y, x] *= y / <float>s

    return E_arr
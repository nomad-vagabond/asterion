import math, warnings
from copy import deepcopy
# import pickle 

import numpy as np
import pandas as pd

import read_database as rdb


def cut_params(hazdf, nohazdf, cutcol):
    """ 
    Cuts columns from the dataframes of PHAs and NHAs specified by 'cutcol'
    parameter and returns list of corresponding arrays.
    """
    data_arr = []
    for dataframe in [hazdf, nohazdf]:
        if dataframe is not None:
            cutdata = dataframe[cutcol]
            arr = cutdata.as_matrix()
        else: arr = None
        data_arr.append(arr)
    return data_arr

def cut_normalize(cutcol, *haz_nohaz_pairs):
    """
    Cuts and normalizes columns from the pairs of dataframes 
    representing PHAs and NHAs.
    """

    pair_cuts = []
    for pair in haz_nohaz_pairs:
        haz_cut, nohaz_cut = cut_params(pair[0], pair[1], cutcol)
        pair_cuts.append([haz_cut, nohaz_cut])

    all_cuts = [cut for pair in pair_cuts for cut in pair]
    bounds = common_bounds(all_cuts)

    cut_scales = []
    for cut in all_cuts:
        cut_norm, sc = normalize_dataset(cut, bounds, copy=False)
        cut_scales.append(sc)
    
    scales = common_scales(cut_scales)

    return pair_cuts, scales

def normalize_dataset(dataset, bounds=None, copy=True):
    """
    Normalizes dataset for the boundary values of each column. Boundary values 
    from the 'bounds' parameter are used unless it is None. Otherwise maximum and
    minimum values for each column are used as boundary values.
    """

    if copy:
        dataset_out = np.zeros_like(dataset)
    else:
        dataset_out = dataset

    if len(dataset.shape) > 1:
        scales = []
        ncol = dataset.shape[1]    
        for col in range(ncol):
            if bounds is None:
                col_min, col_max = np.min(dataset[:, col]), np.max(dataset[:, col])
            else:
                col_min, col_max = bounds[0][col], bounds[1][col]

            scales.append((col_min, col_max))
            scale = col_max - col_min
            dataset_out[:, col] = (dataset[:, col] - col_min)/scale
        # return dataset_out
    else:
        if bounds is None:
            data_min, data_max = np.min(dataset), np.max(dataset)
        else:
            data_min, data_max = bounds[0][0], bounds[1][0]

        scale = data_max - data_min
        scales = [data_min, data_max]
        dataset_out = (dataset - data_min)/scale
        
    return dataset_out, scales

def dmirror_clusters(clusters_, colnum, value):
    """ 
    Returns clusters with mirrored copies of the original cluster points over 
    the 'value' and the half value.
    """

    # clusters_cut = [c[:, :-1] for c in clusters]
    clusters = deepcopy(clusters_)
    clusters_dm = []
    for clust in clusters:
        # clust_cut, clust_id = clust[:, :-1], clust[:, -1]
        mircol = clust[:, colnum]
        all_inds = set(range(len(clust)))
        left_inds = np.where(mircol < value)[0]
        right_inds = list(all_inds - set(left_inds))
        right_inds = np.array(right_inds, dtype=int)

        mircol[left_inds] = value - mircol[left_inds]
        mircol[right_inds] = 3*value - mircol[right_inds]

        clust_m = deepcopy(clust)
        clust_m[:, colnum] = mircol

        clust_e = np.concatenate((clust, clust_m))
        mircol2 = clust_e[:, colnum]
        mircol2 = value*2 - mircol2

        clust_dm = deepcopy(clust_e)
        clust_dm[:, colnum] = mircol2

        clust_e2 = np.concatenate((clust_e, clust_dm))
        clusters_dm.append(clust_e2)
    return clusters_dm

def append_phacol(hazarr, nohazarr):
    """Appends PHA id column to the PHA and NHA arrays."""

    num_haz, num_nohaz = len(hazarr), len(nohazarr)
    phacol = np.reshape(np.ones(num_haz), (num_haz, 1))
    nophacol = np.reshape(np.zeros(num_nohaz), (num_nohaz, 1))

    hazarr_ = np.append(hazarr, phacol, axis=1)
    nohazarr_ = np.append(nohazarr, nophacol, axis=1)
    return hazarr_, nohazarr_

def split_by_colval(dataset, colname, value):
    dataset_left = dataset[dataset[colname] <= value]
    dataset_right = dataset[dataset[colname] > value]
    return dataset_left, dataset_right

def extend_by_copies(dataset, colname, extend_factor=0.5):
    """
    Returns detaset extended by shifted copies of original dataset.
    """

    extendcol = dataset[colname]
    scale = max(extendcol) - min(extendcol)
    cutval_left = scale * extend_factor
    cutval_right = max(extendcol) - scale * extend_factor

    left = deepcopy(dataset[dataset[colname] < cutval_left])
    right = deepcopy(dataset[dataset[colname] > cutval_right])

    left[colname] = left[colname] + scale 
    right[colname] = right[colname] - scale
    data_extend = pd.concat((left, dataset, right))

    return data_extend

def add_doublemirror_column(dataset, colname, value):
    """ 
    Extends dataset by its mirrors over the 'value' and 
    the half 'value' of the 'colname'.
    """

    left, right = split_by_colval(dataset, colname, value)
    left_mir, right_mir = map(deepcopy, [left, right])
    left_mir[colname] = value - left[colname]
    right_mir[colname] = 3*value - right[colname]
    half_mirror = pd.concat((left_mir, right_mir, dataset))

    dataset_mirror = deepcopy(half_mirror)
    dataset_mirror[colname] = value*2 - dataset_mirror[colname]
    dataset_extended = pd.concat((half_mirror, dataset_mirror))

    return dataset_extended

def add_mirror_column(dataset, colname, value):
    """ Extends dataset by its mirror over the 'value' of the 'colname'. """

    dataset_mirror = deepcopy(dataset)
    dataset_mirror[colname] = value*2 - dataset[colname]
    dataset_extended = pd.concat((dataset, dataset_mirror))

    return dataset_extended

def shift_and_mirror(dataset, colname, value):
    """ 
    Extends dataset by its halves flipped over the 'value' of the 'colname'.
    """

    left, right = split_by_colval(dataset, colname, value)
    left_shift, right_shift = map(deepcopy, [left, right])
    dataset_mirror = deepcopy(dataset)

    left_shift[colname] = left_shift[colname] + value
    right_shift[colname] = right_shift[colname] - value

    dataset_mirror[colname] = value*2 - dataset_mirror[colname]
    dataset_extended = pd.concat((dataset, left_shift, right_shift, dataset_mirror))

    return dataset_extended

def mix_up(hazarr, nohazarr):
    """Generates train data from arrays of PHAs and NHAs"""

    hazarr_, nohazarr_ = append_phacol(hazarr, nohazarr)
    join_train = np.concatenate((hazarr_, nohazarr_))
    data_train = np.random.permutation(join_train)
    xdata_train, ydata_train = split_by_lastcol(data_train)
    return xdata_train, ydata_train

def common_scales(scale_sets):
    """
    Returns minimal and maximal values of the scales represented by 'scale_sets'.
    """
    scales = np.concatenate(scale_sets, axis=1)
    scales_ = []
    for col_minmax in scales:
        min_val = min(col_minmax)
        max_val = max(col_minmax)
        scales_.append((min_val, max_val))
    return scales_

def dfcommon_bounds(datasets, cols):
    """
    Returns common boundary values (min and max) for each column across the 'datasets'.
    """
    bounds = []
    for col in cols:
        sc = []
        for db in datasets:
            col_min, col_max = min(db[col]), max(db[col])
            sc += [col_min, col_max]
        bounds.append((min(sc), max(sc)))
    return bounds

def common_bounds(datasets):
    """
    Returns tuple lists representing minimal and maximal values of the data columns.
    """

    ncols = [dataset.shape[1] for dataset in datasets]
    if len(np.unique(ncols)) > 1:
        raise ValueError("number of columns in datasets does not match")

    ncol = ncols[0]
    min_vals = [None]*ncol
    max_vals = [None]*ncol
    for dataset in datasets:
        for col in range(ncol):
            col_min, col_max = np.min(dataset[:, col]), np.max(dataset[:, col])
            comp_min = (col_min, min_vals[col])
            min_vals[col] = min(comp_min) if None not in comp_min else col_min
            max_vals[col] = max(col_max, max_vals[col])
    return (min_vals, max_vals)

def split_by_line(hazdb, nohazdb, line, cols, verbose=True):
    """Splits hazardous and non-hazardous datasets by a line in a 2-dimensional space."""

    haz_cut, nohaz_cut = cut_params(hazdb, nohazdb, cols)
    p1, p2 = line
    rotated = _align_vert(p1, p2, haz_cut, nohaz_cut)
    p1_rot, p2_rot, haz_cut_rot, nohaz_cut_rot = rotated
    split = p1_rot[0]

    haz_right = np.where(haz_cut_rot[:, 0] > split)[0]
    nohaz_right = np.where(nohaz_cut_rot[:, 0] > split)[0]

    haz_left = np.where(haz_cut_rot[:, 0] <= split)[0]
    nohaz_left = np.where(nohaz_cut_rot[:, 0] <= split)[0]

    hazdb_right = hazdb.iloc[haz_right]
    nohazdb_right = nohazdb.iloc[nohaz_right]

    hazdb_left = hazdb.iloc[haz_left]
    nohazdb_left = nohazdb.iloc[nohaz_left]

    floatlen = lambda db: float(len(db))
    haz_left_num, nohaz_left_num = map(floatlen, [haz_left, nohaz_left])
    haz_right_num, nohaz_right_num = map(floatlen, [haz_right, nohaz_right])
    
    left_purity = haz_left_num/(haz_left_num + nohaz_left_num)
    right_purity = haz_right_num/(haz_right_num + nohaz_right_num)
    
    if verbose:
        print "PHA purity of the left subset:", left_purity
        print "PHA purity of the right subset:", right_purity

    return (hazdb_left, nohazdb_left), (hazdb_right, nohazdb_right)

def split_by_lastcol(dataset):
    variables = dataset[:, :-1]
    target = dataset[:, -1]
    return variables, target

def _align_vert(p1, p2, haz_cut, nohaz_cut):
    # a = (p1[1] - p2[1]) / (p1[0] - p2[0])
    # b = p2[1] - a * p2[0]
    
    p1a, p2a = np.asarray(p1), np.asarray(p2)
    v = p2a - p1a
    v0 = -v/np.linalg.norm(v)
    
    x0 = np.array([0,1])
    cosphi = np.dot(v0, x0)
    sinphi = math.sqrt(1 - cosphi**2)
    MR = np.array([[cosphi, -sinphi], [sinphi, cosphi]])
    
    haz_cut_rot = np.asarray([np.dot(hz, MR) for hz in haz_cut])
    nohaz_cut_rot = np.asarray([np.dot(nhz, MR) for nhz in nohaz_cut])
    
    p1_rot = np.dot(p1a, MR)
    p2_rot = np.dot(p2a, MR)
    
    return p1_rot, p2_rot, haz_cut_rot, nohaz_cut_rot






### Deprecated ###

def cut_2params(cols, datasets):
    warnings.warn("this function is deprecated. use cut_params instead")
    datasets = prepare_data(cutcol=cols, datasets=datasets)
    datasets_x = [datasets[i][:, :-1] for i in range(len(datasets))]
    return datasets_x[:2]

def prepare_data(cutcol=['a', 'e'], datasets=None):
    # print "prepare..."
    if datasets is None:
        datasets = map(rdb.loadObject, sources)
    data_arr = []
    ds = int(len(datasets)/2.0)
    bins = [1,0,1,0]*ds
    for pha, dataset in zip(bins, datasets):
        cutdata = dataset[cutcol]
        arr = cutdata.as_matrix()
        phacol = np.array([[pha]*len(arr)]).T
        arr_ = np.append(arr, phacol, axis=1)
        data_arr.append(arr_)
    return data_arr

def learning_sets(haz, nohaz, cutcol):
    haz_cut, nohaz_cut = prepare_data(cutcol=cutcol, datasets=[haz, nohaz])
    merged = np.concatenate((haz_cut, nohaz_cut))
    data = np.random.permutation(merged)
    return split_by_lastcol(data)

### Leftovers ###

# def form_datasets(data):
#     trial_cut = int(len(data)*0.9)
#     data_trial = data[:trial_cut]
#     data_test = data[trial_cut:]
#     # print "data_trial_rand:\n", data_trial_rand[:10]
#     return data_trial, data_test

# def get_learndata(datasets, split=True):
#     # datasets = prepare_data(cutcol)
#     # print "gen..."
#     join_train = np.concatenate((datasets[0], datasets[1]))
#     join_test = np.concatenate((datasets[2], datasets[3]))

#     if split:
#         data_train = np.random.permutation(join_train)
#         data_test = np.random.permutation(join_test)
#         xdata_train, ydata_train = split_by_lastcol(data_train)
#         xdata_test, ydata_test = split_by_lastcol(data_test)
#         # print "xdata_train:", xdata_train[:5]
#         return xdata_train, ydata_train, xdata_test, ydata_test
#     else:
#         mix = np.concatenate((join_train, join_test))
#         permutate = np.random.permutation(mix)
#         xdata, ydata = split_by_lastcol(permutate)
#         return xdata, ydata

# def ncut_params(hazdf, nohazdf, cutcol, bounds=None):
#     import asterion_learn as al
#     haz_cut, nohaz_cut = cut_params(hazdf, nohazdf, cutcol)
#     if bounds is None:
#         bounds = al.common_bounds([haz_cut, nohaz_cut])
#     haz_cut, haz_sc = normalize_dataset(haz_cut, bounds=bounds)
#     nohaz_cut, nohaz_sc = normalize_dataset(nohaz_cut, bounds=bounds)
#     scales = common_scales([haz_sc, nohaz_sc])
#     return haz_cut, nohaz_cut, scales


# if __name__ == '__main__':
#     print "init orbits update..."
#     update_datasets()
#     print "orbits update finished."
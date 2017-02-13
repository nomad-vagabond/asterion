from __future__ import division
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm, cross_validation
from sklearn.cluster import DBSCAN
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.grid_search import GridSearchCV
# from sklearn import cluster
# from sklearn.neighbors.nearest_centroid import NearestCentroid
# from sklearn.gaussian_process import GaussianProcess
# from sklearn import mixture
# from sklearn.preprocessing import normalize

import visualize_data as vd
import learn_data as ld


def sgmask_clf2d_fitcut(clf, inner_cut, outer_cut):
    """
    Fits classifier to separate asteroids belonging to the subgroup 
    from the rest of asteroids. 
    """
    innum = len(inner_cut)
    sgincol = np.reshape(np.ones(innum), (innum, 1))
    inner_cut_id = np.append(inner_cut, sgincol, axis=1)

    outnum = len(outer_cut)
    sgoutcol = np.reshape(np.zeros(outnum), (outnum, 1))
    outer_cut_id = np.append(outer_cut, sgoutcol, axis=1)

    together = np.concatenate((inner_cut_id, outer_cut_id))
    together = np.random.permutation(together)

    xtrain, ytrain = ld.split_by_lastcol(together)
    clf = clf.fit(xtrain, ytrain)

    return clf

def sgmask_clf2d_fit(clf, cutcol, inner, outer, scales):
    """
    Fits classifier to separate asteroids belonging to the subgroup 
    from the rest of asteroids. 
    """

    x, y = cutcol
    xmin, xmax = scales[0]
    ymin, ymax = scales[1]

    inner_c = inner[cutcol]
    outer_c = outer[cutcol]

    inner_c = inner_c[inner_c[x] >= xmin]
    inner_c = inner_c[inner_c[x] <= xmax]
    inner_c = inner_c[inner_c[y] >= ymin]
    inner_c = inner_c[inner_c[y] <= ymax]

    outer_c = outer_c[outer_c[x] >= xmin]
    outer_c = outer_c[outer_c[x] <= xmax]
    outer_c = outer_c[outer_c[y] >= ymin]
    outer_c = outer_c[outer_c[y] <= ymax]

    inner_cut = inner_c.as_matrix()
    outer_cut = outer_c.as_matrix()

    bounds = np.asarray(scales).T

    inner_cut, insc = ld.normalize_dataset(inner_cut, bounds=bounds)
    outer_cut, outsc = ld.normalize_dataset(outer_cut, bounds=bounds)

    innum = len(inner_cut)
    sgincol = np.reshape(np.ones(innum), (innum, 1))
    inner_cut_id = np.append(inner_cut, sgincol, axis=1)

    outnum = len(outer_cut)
    sgoutcol = np.reshape(np.zeros(outnum), (outnum, 1))
    outer_cut_id = np.append(outer_cut, sgoutcol, axis=1)

    together = np.concatenate((inner_cut_id, outer_cut_id))
    together = np.random.permutation(together)

    xtrain, ytrain = ld.split_by_lastcol(together)
    clf = clf.fit(xtrain, ytrain)

    return clf

def clf_split_quality(clf, haz_cut, nohaz_cut, verbose=True):
    """
    Calculates and displays quality of the NEAs division into PHA 
    and NHA regions by classifier.
    """

    haz_clf = clf.predict(haz_cut)
    nohaz_clf = clf.predict(nohaz_cut)
    
    haz_1 = np.where(haz_clf == 1)[0]
    nohaz_1 = np.where(nohaz_clf == 1)[0]
    haz_0 = np.where(haz_clf == 0)[0]
    nohaz_0 = np.where(nohaz_clf == 0)[0]
      
    # floatlen = lambda db: float(len(db))
    haz_1num, nohaz_1num = map(len, [haz_1, nohaz_1])
    haz_0num, nohaz_0num = map(len, [haz_0, nohaz_0])
    
    haz_purity = haz_1num/(haz_1num + nohaz_1num)
    nohaz_purity = nohaz_0num/(haz_0num + nohaz_0num)
    
    if verbose:
        print "purity of PHA region:", haz_purity
        print "number of PHAs in the PHA region:", haz_1num
        print "number of NHAs in the PHA region:", nohaz_1num
        print
        print "purity of NHA region:", nohaz_purity
        print "number of PHAs in the NHA region:", haz_0num
        print "number of NHAs in the NHA region:", nohaz_0num
        print
        print "fraction of correctly classified PHAs:", haz_1num/len(haz_cut)

    return haz_1, nohaz_1, haz_0, nohaz_0
 
def classify_dbclusters(clusters, clf, haz_test, nohaz_test):
    """
    Classifies data by density cluster IDs and calculates PHA mass fraction 
    in the clusters.
    """

    mixed = np.random.permutation(np.concatenate(tuple(clusters)))
    mixed_x, mixed_y = ld.split_by_lastcol(mixed)
    clf = clf.fit(mixed_x, mixed_y)

    predict_haz = clf.predict(haz_test)
    predict_nohaz = clf.predict(nohaz_test)

    classnum_haz = np.bincount(predict_haz.astype(int))
    classnum_nohaz = np.bincount(predict_nohaz.astype(int))

    haz_prob = ([haz/float(haz + nohaz) 
                for haz, nohaz in zip(classnum_haz, classnum_nohaz)])

    return mixed_x, clf, haz_prob

def extract_dbclusters(data, dens_layers, verbose=False):
    """
    Iterratively finds density-based clusters in the data with DBSCAN. Continues serch
    for clusters in the outliers left after the previous itteration if the dens_layers
    parameter contains several values for eps and min_smples.
    """

    data_ = deepcopy(data)
    level = 1
    extracted_clusters = []
    for eps, min_samples in dens_layers:
        # densclust = density_clusters(data_, eps=eps, min_samples=min_samples)
        data_norm, scales = ld.normalize_dataset(data_)
        densclust = DBSCAN(eps=eps, min_samples=min_samples).fit(data_norm)
        max_ind = max(densclust.labels_)

        for i in range(max_ind + 1):
            clusters_ind = np.where(densclust.labels_ == i)[0]
            extracted = data_[clusters_ind]
            id_col = (i + level) * np.ones((len(extracted),1), dtype=int)
            extracted = np.append(extracted, id_col, axis=1)
            extracted_clusters.append(extracted)

        rest_ind = np.where(densclust.labels_ == -1)[0]
        data_ = data_[rest_ind]
        level += (max(densclust.labels_) + 1)

    id_col = (level + 0) * np.ones((len(data_),1), dtype=int)
    extracted = np.append(data_, id_col, axis=1)
    extracted_clusters.append(extracted)
    data_ = []

    if verbose:
        print "extracted_clusters\n [ID, number of elements]"
        for ec in extracted_clusters:
            print [ec[0][-1], len(ec)]
    
    return extracted_clusters

def merge_dbclusters(clusters, megrgemap, merge_rest=True):
    """
    Merges clusters by their IDs specified by the megrgemap (nested list).
    """
    
    clusters_merged = []
    clusters_ = deepcopy(clusters)
    left_inds = set(range(1, len(clusters)+1))
    merged_inds = []

    for clust_ids in megrgemap:
        merged = []
        for i, cluster in enumerate(clusters):   
            if (i+1) in clust_ids:
                merged_inds.append(i+1)
                if len(merged) > 0:
                    merged = np.concatenate((merged, cluster))
                else:
                    merged = cluster
        clusters_merged.append(merged)

    left_inds = left_inds - set(merged_inds)

    if merge_rest:
        rest = []
        for j in list(left_inds):
            cluster = clusters[j-1]
            if len(rest) > 0:
                rest = np.concatenate((rest, cluster))
            else:
                rest = cluster
        clusters_merged.append(rest)
    else:
        for j in list(left_inds):
            cluster = clusters[j-1]
            clusters_merged.append(cluster)

    for i, cluster in enumerate(clusters_merged):
        cluster[...,-1] = i + 1

    return clusters_merged

def split_minigroups(subgroup, levels):
    slices = list(zip(levels[:-1]*0.001, levels[1:]*1.001))
    minigroups = [[] for s in slices]
    minigroups_inds = [[] for s in slices]

    for i, v in enumerate(subgroup):
        for j, s in enumerate(slices):
            if v > s[0] and v <= s[1]:
                minigroups[j].append(v)
                minigroups_inds[j].append(i)
                break

    return minigroups_inds

def normgrid_kde(kde, num=101, levnum=4, scales=[(0,1), (0,1)]):
    grid = np.linspace(0, 1, num)
    X, Y = np.meshgrid(grid, grid)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    Z = kde.score_samples(xy).reshape(X.shape)
    levels = np.linspace(0, Z.max(), levnum)
    X_, Y_ = vd._get_datagrid(scales[0], scales[1], num)

    return levels, [X_, Y_, Z]


### Deprecated ###

def sgmask_clf(hazdf, nohazdf, hazdf_rest, nohazdf_rest, clf, cutcol):
    """
    Fits classifier to separate asteroids belonging to the subgroup 
    from the rest of asteroids. 
    """

    df = pd.concat((hazdf, nohazdf))
    x, y = cutcol[0], cutcol[1]
    xmin, xmax = min(df[x]), max(df[x])
    ymin, ymax = min(df[y]), max(df[y])

    datacut = df[cutcol].as_matrix()
    datacut, scales = ld.normalize_dataset(datacut)

    ndata = len(datacut)
    sgincol = np.reshape(np.ones(ndata), (ndata, 1))
    datacut_ = np.append(datacut, sgincol, axis=1)

    rest = pd.concat((hazdf_rest, nohazdf_rest))
    rest = rest[rest[x] >= xmin]
    rest = rest[rest[x] <= xmax]

    rest = rest[rest[y] >= ymin]
    rest = rest[rest[y] <= ymax]

    restcut = rest[cutcol].as_matrix()
    restcut, scales = ld.normalize_dataset(restcut)
    nrest = len(restcut)
    sgoutcol = np.reshape(np.zeros(nrest), (nrest, 1))
    restcut_ = np.append(restcut, sgoutcol, axis=1)

    data_rest = np.concatenate((datacut_, restcut_))
    data_rest = np.random.permutation(data_rest)

    xtrain, ytrain = ld.split_by_lastcol(data_rest)
    clf = clf.fit(xtrain, ytrain)

    # c = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # c1i = np.where(c==1)[0]
    # c0i = np.where(c==0)[0]

    return clf

def split_by_clf(clf, cutcol, haz_train, nohaz_train, 
                 haz_test=None, nohaz_test=None, verbose=True):
    """
    Splits datasets by classifier. Returns subsets of initial datasets split
    by classifier and scales of each column.
    """

    haz_test = deepcopy(haz_train) if haz_test is None else haz_test
    nohaz_test = deepcopy(nohaz_train) if nohaz_test is None else nohaz_test
    
    haz_train_cut, nohaz_train_cut = ld.cut_params(haz_train, nohaz_train, cutcol)
    haz_test_cut, nohaz_test_cut = ld.cut_params(haz_test, nohaz_test, cutcol)

    bounds = ld.common_bounds([haz_train_cut, nohaz_train_cut, 
                               haz_test_cut, nohaz_test_cut])
    
    haz_train_cut, haz_train_sc = ld.normalize_dataset(haz_train_cut, bounds)
    nohaz_train_cut, nohaz_train_sc = ld.normalize_dataset(nohaz_train_cut, bounds)
    haz_test_cut, haz_test_sc = ld.normalize_dataset(haz_test_cut, bounds)
    nohaz_test_cut, nohaz_test_sc = ld.normalize_dataset(nohaz_test_cut, bounds)

    scales = ld.common_scales([haz_train_sc, nohaz_train_sc, 
                               haz_test_sc, nohaz_test_sc])
    
    xtrain, ytrain = ld.mix_up(haz_train_cut, nohaz_train_cut)
    clf = clf.fit(xtrain, ytrain)
        
    predicted = clf_split_quality(clf, haz_test_cut, nohaz_test_cut, verbose=verbose)
    haz_1, nohaz_1, haz_0, nohaz_0 = predicted

    haz_test_1 = haz_test.iloc[haz_1]
    nohaz_test_1 = nohaz_test.iloc[nohaz_1]
    haz_test_0 = haz_test.iloc[haz_0]
    nohaz_test_0 = nohaz_test.iloc[nohaz_0]

    haz_concat = pd.concat((haz_test_1, nohaz_test_1))
    nohaz_concat = pd.concat((haz_test_0, nohaz_test_0))

#     return haz_concat, nohaz_concat
    return (haz_test_1, nohaz_test_1), (haz_test_0, nohaz_test_0), scales

def merge_clusters(data, labels, class_id, tail=False):
    """Merges all density-based clusters and add class ID column"""
    if tail:
        # clusters_ind = np.where(labels == -1)[0]
        merged = data
        leftovers = []
    else:
        clusters_ind = np.where(labels >= 0)[0]
        rest_ind = np.where(labels == -1)[0]
        leftovers = data[rest_ind]
        merged = data[clusters_ind]

    id_col = class_id*np.ones((len(merged),1), dtype=int)
    merged = np.append(merged, id_col, axis=1)

    # rest_ind = np.where(labels == -1)[0]
    # leftovers = data[rest_ind]

    return merged, leftovers
        
def classify_hazardous(datasets, clf, crossval=False):
    xtrain, ytrain, xtest, ytest = ld.get_learndata(datasets)
    # map(normalize_dataset, [xtrain, xtest])
    # clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    print "clf:", clf
    fit_predict(xtrain, ytrain, xtest, ytest, clf)

    if crossval:
        print "init cross validation..."
        xdata, ydata = ld.get_learndata(datasets, split=False)
        k_fold = cross_validation.KFold(n=len(xdata), n_folds=3)
        for tr, ts in k_fold:
            xtrain, ytrain = xdata[tr], ydata[tr]
            xtest, ytest = xdata[ts], ydata[ts]
            # map(normalize_dataset, [xtrain, xtest])
            xtrain, s_ = ld.normalize_dataset(xtrain)
            xtest, s_ = ld.normalize_dataset(xtest)
            fit_predict(xtrain, ytrain, xtest, ytest, clf)
        print "done."

    # if plotclf:
    #     haz_real, nohaz_real = map(normalize_dataset, 
    #                                [datasets[i][:, :-1] for i in [2,3]])
    #     vd.plot_classifier(xtrain, clf, num=200, haz=haz_real, figsize=figsize,
    #                        nohaz=nohaz_real, labels=['Perihelion distance (q)',
    #                        'Argument of periapsis (w)'])
    return xtrain, clf

def fit_predict(xtrain, ytrain, xtest, ytest, clf):
    fitter = clf.fit(xtrain, ytrain)
    predict = clf.predict(xtest)
    test_num = len(xtest)
    predict_match = (np.array([(predict[i] == ytest[i]) for i in range(test_num)]))
    num_predict_match = np.sum(predict_match)
    predict_haz_fraction = np.sum([(val == 1) for val in predict])/float(test_num)
    true_haz_fraction = np.sum([(val == 1) for val in ytest])/float(test_num)
    score = fitter.score(xtest, ytest)
    print "score:", score
    print "predict_haz_fraction:", predict_haz_fraction
    print "true_haz_fraction:", true_haz_fraction

def classify_clusters(data, clf, haz_test, nohaz_test, dens_layers):
    """Classifies data by density cluster IDs and calculates hazardous
       asteroids' mass fraction in the clusters."""
    data_ = data
    # scales = [(min(data[:, 0]), max(data[:, 0])), (min(data[:, 1]), max(data[:, 1]))]
    # print "scales:", scales
    # labels = (-1)*np.ones(len(data))
    merged_clusters = []
    for class_id, (eps, min_samples) in enumerate(dens_layers):
        densclust = density_clusters(data_, eps=eps, min_samples=min_samples)
        print "len(densclust.labels_):", len(densclust.labels_), type(densclust.labels_)
        print np.unique(densclust.labels_)
        merged, data_ = merge_clusters(data_, densclust.labels_, class_id)
        merged_clusters.append(merged)

    merged, data_ = merge_clusters(data_, densclust.labels_, class_id+1, tail=True)
    merged_clusters.append(merged)

    # vd.plot_densclusters(merged_clusters, scales=scales)
    merged_p = np.random.permutation(np.concatenate(tuple(merged_clusters)))

    # clf = KNeighborsClassifier(n_neighbors=int(0.01*len(data)))
    # clf = svm.SVC(C=1, gamma=100.) #kernel='poly'
    merged_px, merged_py = ld.split_by_lastcol(merged_p)
    fitter = clf.fit(merged_px, merged_py)

    ids = range(len(dens_layers)+1)
    predict_haz = clf.predict(haz_test)
    predict_nohaz = clf.predict(nohaz_test)
    # print "predict_haz:", predict_haz[:10]
    # print "predict_nohaz:", predict_nohaz[:10]

    classnum_haz = np.bincount(predict_haz.astype(int))
    classnum_nohaz = np.bincount(predict_nohaz.astype(int))
    # print "classnum_haz:", classnum_haz[:10]
    # print "classnum_nohaz:", classnum_nohaz[:10]

    haz_prob = ([haz/float(haz + nohaz) 
                for haz, nohaz in zip(classnum_haz, classnum_nohaz)])
    print "haz_prob:", haz_prob

    # scales = [(min(data[:, 0]), max(data[:, 0])), (min(data[:, 1]), max(data[:, 1]))]
    # vd.plot_classifier(merged_px, clf, num=200, haz=haz_test, nohaz=nohaz_test, clustprobs=haz_prob, scales=scales)
    return merged_clusters, merged_px, clf, haz_prob

def density_clusters(data_x, eps=0.015, min_samples=100, plotclusters=False, 
                     figsize=(10, 10)):

    """
    Finds density-based clusters in data with DBSCAN.
    """
    data_x_norm, scales = ld.normalize_dataset(data_x)
    dbsc = DBSCAN(eps=eps, min_samples=min_samples).fit(data_x_norm)  # 0.015  100  | 0.021  160
    labels = dbsc.labels_
    unique_labels = np.unique(labels)
    core_samples = np.zeros_like(labels, dtype = bool)
    core_samples[dbsc.core_sample_indices_] = True
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print "n_clusters_:", n_clusters_
    # print "core_samples:", core_samples[:50]
    # colors = ['yellow', 'red', 'green', 'blue', 'magenta']
    colors_ = vd.get_colorlist(len(unique_labels))
    # print "labels:", labels, len(labels)
    # print "unique_labels:", unique_labels

    if plotclusters:
        fig = plt.figure(figsize=figsize)
        for (label, color) in zip(unique_labels, colors_):
            if label == -1:
                color = "white"
            class_member_mask = (labels == label)
            xy = data_x_norm[class_member_mask & core_samples]
            plt.plot(xy[:,0],xy[:,1], 'o', markerfacecolor = color, markersize = 5)
            
            xy2 = data_x_norm[class_member_mask & ~core_samples]
            plt.plot(xy2[:,0],xy2[:,1], 'o', markerfacecolor = color, markersize = 4)
        plt.show()
    return dbsc
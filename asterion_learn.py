from __future__ import division
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.colors as colors
# import matplotlib.cm as cmx

from functools import partial
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestNeighbors
from sklearn import svm
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
# from sklearn.grid_search import GridSearchCV
from sklearn import cluster
# from sklearn.neighbors.nearest_centroid import NearestCentroid
# from sklearn.gaussian_process import GaussianProcess
# from sklearn import mixture
from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import normalize


# from draw_ellipse_3d import OrbitDisplayGL
# import pickle
import visualize_data as vd
from learn_data import get_learndata, prepare_data, split_by_lastcol
import learn_data as ld
from read_database import loadObject, dumpObject
# reload(ld)

def sgmask_clf2d(clf, inner, outer):
    """
    Fits classifier to separate asteroids belonging to the subgroup 
    from the rest of asteroids. 
    """
    innum = len(inner)
    sgincol = np.reshape(np.ones(innum), (innum, 1))
    inner_ = np.append(inner, sgincol, axis=1)

    outnum = len(outer)
    sgoutcol = np.reshape(np.zeros(outnum), (outnum, 1))
    outer_ = np.append(outer, sgoutcol, axis=1)

    together = np.concatenate((inner_, outer_))
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
        
def classify_hazardous(datasets, clf, crossval=False):
    xtrain, ytrain, xtest, ytest = get_learndata(datasets)
    # map(normalize_dataset, [xtrain, xtest])
    # clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    print "clf:", clf
    fit_predict(xtrain, ytrain, xtest, ytest, clf)

    if crossval:
        print "init cross validation..."
        xdata, ydata = get_learndata(datasets, split=False)
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

def density_clusters(data_x, eps=0.015, min_samples=100, plotclusters=False, 
                     figsize=(10, 10)):
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

def classify_clusters(data, clf, haz_test, nohaz_test, dens_layers):
    """Classifies data by density cluster IDs and calculates hazardous
       asteroids' mass fraction in clusters."""
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
    merged_px, merged_py = split_by_lastcol(merged_p)
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
 
def classify_dbclusters(clusters, clf, haz_test, nohaz_test):
    """Classifies data by density cluster IDs and calculates hazardous
       asteroids' mass fraction in clusters."""

    mixed = np.random.permutation(np.concatenate(tuple(clusters)))

    # clf = KNeighborsClassifier(n_neighbors=int(0.01*len(data)))
    # clf = svm.SVC(C=1, gamma=100.) #kernel='poly'
    mixed_x, mixed_y = split_by_lastcol(mixed)
    # print "np.unique(mixed_y):", np.unique(mixed_y)
    # print np.bincount(mixed_y.astype(int))
    # print
    clf = clf.fit(mixed_x, mixed_y)

    # ids = range(1, len(clusters)+1)
    # print "ids:", ids
    predict_haz = clf.predict(haz_test)
    predict_nohaz = clf.predict(nohaz_test)
    # print "predict_haz:", predict_haz[:10]
    # print "predict_nohaz:", predict_nohaz[:10]
    # print
    # print "np.unique(predict_haz):", np.unique(predict_haz)
    # print "np.unique(predict_nohaz):", np.unique(predict_nohaz)

    classnum_haz = np.bincount(predict_haz.astype(int))
    classnum_nohaz = np.bincount(predict_nohaz.astype(int))
    # print "classnum_haz:", classnum_haz
    # print "classnum_nohaz:", classnum_nohaz

    haz_prob = ([haz/float(haz + nohaz) 
                for haz, nohaz in zip(classnum_haz, classnum_nohaz)])
    # print "haz_prob:", haz_prob

    # scales = [(min(data[:, 0]), max(data[:, 0])), (min(data[:, 1]), max(data[:, 1]))]
    # vd.plot_classifier(merged_px, clf, num=200, haz=haz_test, nohaz=nohaz_test, clustprobs=haz_prob, scales=scales)
    return mixed_x, clf, haz_prob

def extract_dbclusters(data, dens_layers, verbose=False):
    data_ = deepcopy(data)
    level = 1
    extracted_clusters = []
    for eps, min_samples in dens_layers:
        densclust = density_clusters(data_, eps=eps, min_samples=min_samples)
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
    
    clusters_merged = []
    clusters_ = deepcopy(clusters)
    left_inds = set(range(1, len(clusters)+1))
    # print "left_inds before:", left_inds
    merged_inds = []

    for clust_ids in megrgemap:
        merged = []
        # left = []
        # left_inds = set()
        for i, cluster in enumerate(clusters):   
            if (i+1) in clust_ids:
                # left.append(cluster) 
                # left_inds.add(i)
            # else:
                # left_indsG = left_indsG - set([i+1])
                merged_inds.append(i+1)
                if len(merged) > 0:
                    merged = np.concatenate((merged, cluster))
                else:
                    merged = cluster
        # clusters_ = left
        # left_indsG = set(list(left_indsG) + list(left_inds))
        # left.append(merged)
        clusters_merged.append(merged)
    # d = [clusters_merged.append(clust_left) for clust_left in left]
    # print "merged_inds:", merged_inds
    left_inds = left_inds - set(merged_inds)
    # print "len(clusters_merged):", len(clusters_merged)
    # print "left_inds:", left_inds

    if merge_rest:
        rest = []
        # rest = np.hstack(list(left_inds))
        for j in list(left_inds):
            # rest = np.concatenate((rest, clusters[j-1]))
            cluster = clusters[j-1]
            # print "cluster.shape:", cluster.shape
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
    # print "slices:", slices
    minigroups = [[] for s in slices]
    minigroups_inds = [[] for s in slices]
#     print minigroups
    for i, v in enumerate(subgroup):
        for j, s in enumerate(slices):
            if v > s[0] and v <= s[1]:
#                 print "s:", s, v
                minigroups[j].append(v)
                minigroups_inds[j].append(i)
                break
#     return minigroups
    return minigroups_inds

def normgrid_kde(kde, num=101, levnum=4, scales=[(0,1), (0,1)]):
    grid = np.linspace(0, 1, num)
    X, Y = np.meshgrid(grid, grid)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    Z = kde.score_samples(xy).reshape(X.shape)
    levels = np.linspace(0, Z.max(), levnum)
    X_, Y_ = vd._get_datagrid(scales[0], scales[1], num)
    return levels, [X_, Y_, Z]




### Deprecated ##


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
    
    # haz_clf = clf.predict(haz_test_cut)
    # nohaz_clf = clf.predict(nohaz_test_cut)
    
    # haz_1 = np.where(haz_clf == 1)[0]
    # nohaz_1 = np.where(nohaz_clf == 1)[0]
    # haz_0 = np.where(haz_clf == 0)[0]
    # nohaz_0 = np.where(nohaz_clf == 0)[0]
    
    # haz_test_1 = haz_test.iloc[haz_1]
    # nohaz_test_1 = nohaz_test.iloc[nohaz_1]
    # haz_test_0 = haz_test.iloc[haz_0]
    # nohaz_test_0 = nohaz_test.iloc[nohaz_0]
    
    # # floatlen = lambda db: float(len(db))
    # haz_test_1num, nohaz_test_1num = map(len, [haz_test_1, nohaz_test_1])
    # haz_test_0num, nohaz_test_0num = map(len, [haz_test_0, nohaz_test_0])
    
    # haz_purity = haz_test_1num/(haz_test_1num + nohaz_test_1num)
    # nohaz_purity = nohaz_test_0num/(haz_test_0num + nohaz_test_0num)
    
    # if verbose:
    #     print "purity of PHA region:", haz_purity
    #     print "number of PHAs in the PHA region:", haz_test_1num
    #     print "number of NHAs in the PHA region:", nohaz_test_1num
    #     print
    #     print "purity of NHA region:", nohaz_purity
    #     print "number of PHAs in the NHA region:", haz_test_0num
    #     print "number of NHAs in the NHA region:", nohaz_test_0num
    #     print
    #     print "fraction of correctly classified PHAs:",
    #     print haz_test_1num/len(haz_test)
    
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


# def estimate_clusters(data, haz_data, nohaz_data, labels):
#     data_norm = normalize_dataset(data, copy=True)
#     # print "labels:", len(labels)
#     # print "data:", len(data)
#     clusters_ind = np.where(labels >= 0)[0]
#     rest_ind = np.where(labels == -1)[0]
#     # print "clusters_ind:", clusters_ind[:10], len(clusters_ind)
#     # print "rest_ind:", rest_ind[:10], len(rest_ind) 
#     merged = data_norm[clusters_ind]
#     leftovers = data_norm[rest_ind]
#     # print "leftovers:", len(leftovers)
#     ones_col = np.ones((len(merged),1), dtype=int)
#     zeros_col = np.zeros((len(leftovers),1), dtype=int)

#     merged_ = np.append(merged, ones_col, axis=1)
#     leftovers_ = np.append(leftovers, zeros_col, axis=1)

#     vd.plot_distribution(haz=merged, nohaz=leftovers)

#     # cluster_train = np.random.permutation(np.concatenate((merged_, leftovers_)))
#     cluster_train = np.concatenate((merged_, leftovers_))
#     # print "cluster_train:", cluster_train[:10]

#     xtrain, ytrain = split_by_lastcol(cluster_train)

#     # print "xtrain:", xtrain[:10], len(xtrain)
#     # print "ytrain:", ytrain[:10], len(ytrain)
#     # print "np.unique(ytrain):", np.unique(ytrain)

#     # xhaz, yhaz = split_by_lastcol(haz_data)
#     # xnohaz, ynohaz = split_by_lastcol(nohaz_data)

#     clf = KNeighborsClassifier(n_neighbors=10)
#     fitter = clf.fit(xtrain, ytrain)

#     haz_data_, nohaz_data_ = map(normalize_dataset, [haz_data, nohaz_data])

#     vd.plot_classifier(xtrain, clf, num=200, haz=haz_data_, nohaz=nohaz_data_)

#     predict_haz = clf.predict(haz_data_)
#     predict_nohaz = clf.predict(nohaz_data_)

#     # print "predict_haz:\n", predict_haz[:10]
#     # print "predict_nohaz:\n", predict_nohaz[:10]

#     haz_clust = predict_haz[predict_haz == 1]
#     nohaz_clust = predict_nohaz[predict_nohaz == 1]

#     print "len haz_clust:", len(haz_clust)
#     print "len nohaz_clust:", len(nohaz_clust)
#     claster_score = len(haz_clust)/float(len(haz_clust) + len(nohaz_clust))
#     print "claster_score:", claster_score


#     return clf, rest_ind




# def adaboost_predict(trial_x, trial_y, test_x, test_y):
#     clf = AdaBoostClassifier()
#     print "clf:", clf
#     fitter = clf.fit(trial_x, trial_y)
#     score = fitter.score(test_x, test_y)
#     print "score:", score
    
#     predict = clf.predict(test_x)
#     # print "predict:", predict
#     return predict


# def estimate(trial_x, trial_y, test_x, test_y):

#     predict = knn_predict(trial_x, trial_y, test_x, test_y)
#     # predict = svc_predict(trial_x, trial_y, test_x, test_y)
#     # predict = adaboost_predict(trial_x, trial_y, test_x, test_y)
#     # predict = rand_forest_predict()
#     # predict = grid_svc_predict()
#     # predict = crossval_svc_predict(test_x, test_y)

#     test_num = len(test_y)
#     print "test_num:", test_num
#     print "len(predict):", len(predict)

#     predict_match = np.array([(predict[i] == test_y[i]) for i in range(test_num)])
#     num_predict_match = np.sum(predict_match)

#     predict_haz_fraction = np.sum([(val == 1) for val in predict])/float(test_num)
#     true_haz_fraction = np.sum([(val == 1) for val in test_y])/float(test_num)

#     print "predict_haz_fraction:", predict_haz_fraction
#     print "true_haz_fraction:", true_haz_fraction
#     print

#     print "len(predict):", len(predict)
#     print "predict_match:", predict_match[:10], num_predict_match
#     predict_accuracy = 1 - float(len(predict) - num_predict_match)/len(predict)
#     print "predict_accuracy:", predict_accuracy


# def crossval_svc_predict(xdata, ydata):
# # svc = svm.SVC(C=1, kernel='linear')
# # svc = svm.SVC(C=1)
# svc = KNeighborsClassifier(weights='distance', algorithm='kd_tree', n_neighbors=5)
# k_fold = cross_validation.KFold(n=len(xdata), n_folds=3)
# print
# for train_i, test_i in k_fold:
#     xtrain = xdata[train_i]
#     ytrain = ydata[train_i]
#     xtest = xdata[test_i]
#     ytest = ydata[test_i]
#     # print "xtrain:\n", xtrain[:6]
#     # print "ytrain:\n", ytrain[:6]
#     # print "xtest:\n", xtest[:6]
#     # print "ytest:\n", ytest[:6]


#     fitter = svc.fit(xtrain, ytrain)
#     score = fitter.score(xtest, ytest)
#     print "score:", score
#     predict = svc.predict(xtest)
#     test_num = len(ytest)
#     predict_match = np.array([(predict[i] == ytest[i]) for i in range(test_num)])
#     num_predict_match = np.sum(predict_match)

#     predict_haz_fraction = np.sum([(val == 1) for val in predict])/float(test_num)
#     true_haz_fraction = np.sum([(val == 1) for val in ytest])/float(test_num)
#     print "predict_haz_fraction:", predict_haz_fraction
#     print "true_haz_fraction:", true_haz_fraction
# # return predict

# def clf_mask(hazdf, nohazdf, clf, cutcol, bgnum):

#     haz_cut, nohaz_cut = ld.cut_params(hazdf, nohazdf, cutcol)
#     bounds = ld.common_bounds([haz_cut, nohaz_cut])
#     haz_cut, haz_sc = ld.normalize_dataset(haz_cut, bounds)
#     nohaz_cut, nohaz_sc = ld.normalize_dataset(nohaz_cut, bounds)
#     data = np.concatenate(haz_cut, nohaz_cut)
#     ndata = len(data)
#     phacol = np.reshape(np.ones(ndata), (ndata, 1))
#     data_ = np.append(data, phacol, axis=1)

#     xb = yb = [0.0, 1.0]
#     xx, yy = ld._get_datagrid(xb, yb, num)
#     x_ = xx.ravel()
#     y_ = yy.ravel()
#     xcol = np.reshape(x_, (bgnum, 1))
#     ycol = np.reshape(y_, (bgnum, 1))
#     bgcol = np.reshape(np.zeros(bgnum), (bgnum, 1))
#     bg = np.concatenate((xcol, ycol, bgcol), axis=1)

#     data_bg = np.concatenate((data_, bg))
#     data_bg = np.random.permutation(data_bg)

#     xtrain, ytrain = ld.split_by_lastcol(data_bg)
#     clf = clf.fit(xtrain, ytrain)

#     # c = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     # c1i = np.where(c==1)[0]
#     # c0i = np.where(c==0)[0]

#     return clf


# def get_cmap(n):
#     color_norm  = colors.Normalize(vmin=0, vmax=n-1)
#     scalar_map = cm.ScalarMappable(norm=color_norm, cmap='hsv') 
#     colors_list = [scalar_map.to_rgba(index) for index in range(n)]
#     return colors_list
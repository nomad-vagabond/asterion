import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cmx

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestNeighbors
# from sklearn import svm
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
from read_database import loadObject, dumpObject


def get_cmap(n):
    color_norm  = colors.Normalize(vmin=0, vmax=n-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    colors_list = [scalar_map.to_rgba(index) for index in range(n)]
    return colors_list

def classify_knn(datasets, n_neighbors=500, crossval=False, plotclf=True,
                 figsize=(10, 10)):
    xtrain, ytrain, xtest, ytest = get_learndata(datasets)
    map(normalize_dataset, [xtrain, xtest])
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    print "clf:", clf
    fit_predict(xtrain, ytrain, xtest, ytest, clf)

    if crossval:
        print "init cross validation..."
        xdata, ydata = get_learndata(datasets, split=False)
        k_fold = cross_validation.KFold(n=len(xdata), n_folds=3)
        for tr, ts in k_fold:
            xtrain, ytrain = xdata[tr], ydata[tr]
            xtest, ytest = xdata[ts], ydata[ts]
            map(normalize_dataset, [xtrain, xtest])
            fit_predict(xtrain, ytrain, xtest, ytest, clf)
        print "done."

    if plotclf:
        haz_real, nohaz_real = map(normalize_dataset, 
                                   [datasets[i][:, :-1] for i in [2,3]])
        vd.plot_classifier(xtrain, clf, num=200, haz=haz_real, figsize=figsize,
                           nohaz=nohaz_real, labels=['Perihelion distance (q)',
                           'Argument of periapsis (w)'])
    return clf

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

def density_clusters(data_x, eps=0.015, min_samples=100, plotclusters=True, 
                     figsize=(10, 10)):
    data_x_norm = normalize_dataset(data_x, copy=True)
    dbsc = DBSCAN(eps=eps, min_samples=min_samples).fit(data_x_norm)  # 0.015  100  | 0.021  160
    labels = dbsc.labels_
    unique_labels = np.unique(labels)
    core_samples = np.zeros_like(labels, dtype = bool)
    core_samples[dbsc.core_sample_indices_] = True
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print "n_clusters_:", n_clusters_
    # print "core_samples:", core_samples[:50]
    # colors = ['yellow', 'red', 'green', 'blue', 'magenta']
    colors_ = get_cmap(len(unique_labels))
    # print "labels:", labels, len(labels)
    print "unique_labels:", unique_labels

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

def normalize_dataset(dataset, copy=False):
    ncol = dataset.shape[1]
    if copy:
        dataset_out = np.zeros_like(dataset)
    else:
        dataset_out = dataset
    for col in range(ncol):
        col_min, col_max = np.min(dataset[:, col]), np.max(dataset[:, col])
        scale = col_max - col_min
        dataset_out[:, col] = (dataset[:, col] - col_min)/scale
    return dataset_out

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

    
def multiclass_knn(data, haz_test, nohaz_test, dens_layers):
    """Classifies data by density cluster IDs and calculates hazardous
       asteroids' mass fraction in clusters."""
    data_ = data
    # labels = (-1)*np.ones(len(data))
    merged_clusters = []

    for class_id, (eps, min_samples) in enumerate(dens_layers):
        densclust = density_clusters(data_, eps=eps, min_samples=min_samples)
        merged, data_ = merge_clusters(data_, densclust.labels_, class_id)
        merged_clusters.append(merged)

    merged, data_ = merge_clusters(data_, densclust.labels_, class_id+1, tail=True)
    merged_clusters.append(merged)
    merged_p = np.random.permutation(np.concatenate(tuple(merged_clusters)))

    clf = KNeighborsClassifier(n_neighbors=10)
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

    vd.plot_classifier(merged_px, clf, num=300)
    return haz_prob



if __name__ == '__main__':

    ### LOAD DATASETS ###
    datasets = prepare_data(cutcol=['q', 'w'])
    # xdata_train, ydata_train, xdata_test, ydata_test = get_learndata(datasets)

    ### DISPLAY PARAMETERS DISTRIBUTION ###
    datasets_x = [datasets[i][:, :-1] for i in range(4)]
    haz_gen, nohaz_gen, haz_real, nohaz_real = datasets_x
    vd.plot_distribution(haz=haz_gen, nohaz=nohaz_gen, 
                         labels=['Perihelion distance (q)', 
                         'Argument of periapsis (w)'])

    ### NORMALIZE DATASET'S DIMENSIONS ###
    # haz_gen_norm = normalize_dataset(haz_gen, copy=True)
    # print "dataset_haz_gen_norm:", dataset_haz_gen_norm[:10]

    ## CLASSIFY DATA WITH KNN ###
    classify_knn(datasets, n_neighbors=500, crossval=True)

    ## CLASSIFY DATA WITH KNN USING DENSITY CLUSTERING ###
    map(normalize_dataset, [haz_gen, haz_real, nohaz_real])
    eps = [0.0188, 0.02, 0.027, 0.04]
    min_samples = [275, 220, 140, 100]
    dens_layers = zip(eps, min_samples)
    multiclass_knn(haz_gen, haz_real, nohaz_real, dens_layers)





    # ### FIND DENSITY-BASED CLUSTERS ###
    # densclust = density_clusters(haz_gen, eps=0.0188, min_samples=275)  # eps=0.016, min_samples=180 200


    # # clusters = range(n_clusters)

    # labels = densclust.labels_

    # haz_clust, nohaz_clust, rest_ind = estimate_clusters(haz_gen, haz_real, nohaz_real, labels) 
    # # haz_real, nohaz_real | haz_gen, nohaz_gen

    # clusters_ind2 = np.where(labels == -1)
    # haz_gen2 = haz_gen[clusters_ind2]
    # densclust, n_clusters = density_clusters(haz_gen2, eps=0.023, min_samples=150)
    # labels2 = densclust.labels_
    # haz_clust2, nohaz_clust2, rest_ind2 = estimate_clusters(haz_gen2, haz_real, nohaz_real, labels2) 
    # haz_real, nohaz_real | haz_gen, nohaz_gen


    # estimate(xdata_train, ydata_train, xdata_test, ydata_test)

    # xdata, ydata = get_learndata(datasets, split=False)
    # crossval_svc_predict(xdata, ydata)
    # disp_orbit.show()
    # clf = loadObject('classifier.p')
    # print clf.predict(np.array([[0.4, 0.6]]))[0]

    # dumpObject(clf, 'classifier.p')


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
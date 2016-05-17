import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cmx

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
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
from learn_data import get_learndata, prepare_data
from read_database import loadObject, dumpObject


def get_cmap(n):
    color_norm  = colors.Normalize(vmin=0, vmax=n-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    colors_list = [scalar_map.to_rgba(index) for index in range(n)]
    return colors_list

def classify_knn(datasets, n_neighbors=500, crossval=False, plotclf=True,
                 figsize=(10, 10)):
    xdata_train, ydata_train, xdata_test, ydata_test = get_learndata(datasets)
    map(normalize_dataset, [xdata_train, xdata_test])
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    print "clf:", clf

    # if crossval:
    #     xdata = 
    #     k_fold = cross_validation.KFold(n=len(xdata), n_folds=3)
    #     for train_i, test_i in k_fold:


    fitter = clf.fit(xdata_train, ydata_train)
    predict = clf.predict(xdata_test)
    test_num = len(xdata_test)
    predict_match = (np.array([(predict[i] == ydata_test[i]) 
                     for i in range(test_num)]))
    num_predict_match = np.sum(predict_match)
    predict_haz_fraction = np.sum([(val == 1) for val in predict])/float(test_num)
    true_haz_fraction = np.sum([(val == 1) for val in ydata_test])/float(test_num)
    score = fitter.score(xdata_test, ydata_test)
    print "score:", score
    print "predict_haz_fraction:", predict_haz_fraction
    print "true_haz_fraction:", true_haz_fraction
    if plotclf:
        haz_real, nohaz_real = map(normalize_dataset, 
                                   [datasets[i][:, :-1] for i in [2,3]])

        vd.plot_classifier(xdata_train, clf, num=100, haz=haz_real, figsize=figsize,
                           nohaz=nohaz_real, labels=['Perihelion distance (q)',
                           'Argument of periapsis (w)'])
    return clf

def density_clusters(data_x, eps=0.015, min_samples=100, plotclusters=True, figsize=(10, 10)):
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

    ### FIND DENSITY-BASED CLUSTERS ###
    density_clusters(haz_gen, eps=0.018, min_samples=140)


    ### CLASSIFY DATA WITH KNN ###
    classify_knn(datasets, n_neighbors=500)


    # estimate(xdata_train, ydata_train, xdata_test, ydata_test)

    # xdata, ydata = get_learndata(datasets, split=False)
    # crossval_svc_predict(xdata, ydata)
    # disp_orbit.show()
    # clf = loadObject('classifier.p')
    # print clf.predict(np.array([[0.4, 0.6]]))[0]

    # dumpObject(clf, 'classifier.p')







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
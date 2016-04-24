import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn import cluster
from sklearn.neighbors.nearest_centroid import NearestCentroid

# import scipy.optimize as so
# from draw_ellipse_3d import OrbitDisplayGL
# import pickle
import visualize_data as vd
# from read_database import calc_rascend, calc_orbc, calc_rclose 
from learn_data import get_learndata, prepare_data
from read_database import loadObject, dumpObject



# def split_by_lastcol(dataset):
#     variables = dataset[:, :-1]
#     target = dataset[:, -1]
#     return variables, target


# def rand_forest_predict():
#     rf = RandomForestClassifier(class_weight="balanced", max_leaf_nodes=15)
#     print "rf:", rf
#     fitter = rf.fit(trial_x, trial_y)
#     score = fitter.score(test_x, test_y)
#     print "score:", score
#     predict = rf.predict(test_x)
#     return predict

# def knn_predict(trial_x, trial_y, test_x, test_y):
#     ### KNeighbors ###
#     knn = KNeighborsClassifier(weights='distance', algorithm='auto', n_neighbors=6)
#     # knn = RadiusNeighborsClassifier() # weights='distance
#     # knn = KNeighborsClassifier(algorithm='auto')
#     print "knn:", knn
#     fitter = knn.fit(trial_x, trial_y)
#     score = fitter.score(test_x, test_y)
#     print "score:", score
#     predict = knn.predict(test_x)
#     return predict

# def svc_predict(trial_x, trial_y, test_x, test_y):
#     # clf = svm.SVC(C=1, kernel='linear')
#     clf = svm.SVC()
#     # clf = svm.NuSVC()
#     # clf = svm.LinearSVC() # multi_class='crammer_singer'
#     print "clf:", clf
#     fitter = clf.fit(trial_x, trial_y)
#     score = fitter.score(test_x, test_y)
#     print "score:", score
    
#     predict = clf.predict(test_x)
#     # print "predict:", predict
#     return predict

def crossval_svc_predict(xdata, ydata):
    # svc = svm.SVC(C=1, kernel='linear')
    # svc = svm.SVC(C=1)
    svc = KNeighborsClassifier(weights='distance', algorithm='kd_tree', n_neighbors=5)
    k_fold = cross_validation.KFold(n=len(xdata), n_folds=3)
    print
    for train_i, test_i in k_fold:
        xtrain = xdata[train_i]
        ytrain = ydata[train_i]
        xtest = xdata[test_i]
        ytest = ydata[test_i]
        # print "xtrain:\n", xtrain[:6]
        # print "ytrain:\n", ytrain[:6]
        # print "xtest:\n", xtest[:6]
        # print "ytest:\n", ytest[:6]


        fitter = svc.fit(xtrain, ytrain)
        score = fitter.score(xtest, ytest)
        print "score:", score
        predict = svc.predict(xtest)
        test_num = len(ytest)
        predict_match = np.array([(predict[i] == ytest[i]) for i in range(test_num)])
        num_predict_match = np.sum(predict_match)

        predict_haz_fraction = np.sum([(val == 1) for val in predict])/float(test_num)
        true_haz_fraction = np.sum([(val == 1) for val in ytest])/float(test_num)
        print "predict_haz_fraction:", predict_haz_fraction
        print "true_haz_fraction:", true_haz_fraction
    # return predict

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



if __name__ == '__main__':


    datasets = prepare_data(cutcol=['q', 'w'])

    # for data in datasets:
    #     print "data:\n", data[:5]
    #     print data.shape

    dataset_haz_gen = datasets[0][:, :-1]
    dataset_nohaz_gen = datasets[1][:, :-1] 
    dataset_haz_real = datasets[2][:, :-1]
    dataset_nohaz_real = datasets[3][:, :-1]
    # # print "type(dataset_haz_gen):", type(dataset_haz_gen)
    vd.plot_distribution(haz=dataset_haz_gen, nohaz=dataset_nohaz_gen, 
                         labels=['Perihelion distance (q)', 'Argument of periapsis (w)'])

    xdata_train, ydata_train, xdata_test, ydata_test = get_learndata(datasets)
    # print "xdata_train:", len(xdata_train), xdata_train.shape
    # print "ydata_train:", len(ydata_train), ydata_train.shape
    # print "xdata_test:", len(xdata_test), xdata_test.shape
    # print "ydata_test:", len(ydata_test), ydata_test.shape

    # clf = KNeighborsClassifier(weights='distance', algorithm='auto', n_neighbors=3)
    # clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
    #                          algorithm="SAMME",
    #                          n_estimators=300)

    # clf = AdaBoostClassifier(n_estimators=300)

    # clf = GradientBoostingClassifier(n_estimators=500, learning_rate=0.9, 
    #min_samples_split=4, min_samples_leaf=8)
    # cw = {0:1, 1:5}
    # clf = svm.SVC(C=100, gamma=100, tol=0.1, kernel='rbf', class_weight=cw, shrinking=False)
    clf = RandomForestClassifier(class_weight="balanced", max_leaf_nodes=500, 
                                 n_estimators=50, criterion='entropy') # class_weight="balanced"
    # clf = RandomForestClassifier(class_weight="balanced_subsample", 
    # max_leaf_nodes=800, n_estimators=300, criterion='entropy', max_features="auto") # class_weight="balanced"
    # clf = DecisionTreeClassifier()
    # clf = RadiusNeighborsClassifier() # weights='distance
    # clf = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=10, weights='distance', leaf_size=200)

    # C_coefs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    # sigma_coefs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    # parameters = {'C':C_coefs, 'gamma': map(sigma_to_gamma, sigma_coefs)}

    # clf = NearestCentroid()
    print "clf:", clf
    fitter = clf.fit(xdata_train, ydata_train)
    score = fitter.score(xdata_test, ydata_test)
    print "score:", score
    predict = clf.predict(xdata_test)

    vd.plot_classifier(xdata_train, clf, num=5e2, haz=dataset_haz_real, nohaz=dataset_nohaz_real,
                       labels=['Perihelion distance (q)', 'Argument of periapsis (w)'])
    # vo.plot_classifier(xdata_train, clf, num=5e2, haz=dataset_haz_gen, nohaz=dataset_nohaz_gen)
    # print "xdata_train:", xdata_train[:5]
    # print "len(xdata_train):", len(xdata_train)
    # print "ydata_train:", ydata_train[:5]
    # print "len(ydata_train):", len(ydata_train)
    # print "xdata_test:", xdata_test[:5]
    # print "len(xdata_test):", len(xdata_test)
    # print "ydata_test:", ydata_test[:5]
    # print "len(ydata_test):", len(ydata_test)

    # estimate(xdata_train, ydata_train, xdata_test, ydata_test)

    # xdata, ydata = get_learndata(datasets, split=False)
    # crossval_svc_predict(xdata, ydata)
    # disp_orbit.show()
    # print clf.predict(np.array([[0.4, 0.6]]))

    dumpObject(clf, 'classifier.p')







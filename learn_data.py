import numpy as np
# from generate_random_orbits import dumpObject, loadObject
# from read_database import calc_rascend, calc_orbc, calc_rclose 
import read_database as rdb
# from functools import partial
import pickle 

# sources = ['./asteroid_data/haz_rand_1e5.p', 
#            './asteroid_data/nohaz_rand_1e5.p',
#            './asteroid_data/haz.p', 
#            './asteroid_data/nohaz.p']

sources = ['./asteroid_data/haz_rand_test.p', 
           './asteroid_data/nohaz_rand_test.p',
           './asteroid_data/haz_test.p', 
           './asteroid_data/nohaz_test.p']


# sources = ['./asteroid_data/haz_rand_test.p', 
#            './asteroid_data/nohaz_rand_test.p']

# def loadObject(fname):
#     obj_file = open(fname,'r')
#     obj = pickle.load(obj_file)
#     obj_file.close()
#     return obj

# def dumpObject(obj, fname):
#     obj_file = open(fname,'wb')
#     pickle.dump(obj, obj_file)
#     obj_file.close()

def form_datasets(data):
    trial_cut = int(len(data)*0.9)
    data_trial = data[:trial_cut]
    data_test = data[trial_cut:]
    # print "data_trial_rand:\n", data_trial_rand[:10]
    return data_trial, data_test

def split_by_lastcol(dataset):
    variables = dataset[:, :-1]
    target = dataset[:, -1]
    return variables, target

def get_wir_points(data):
    for row in data:
        a, e, i, w, omega, q = row
        r = get_r(a, e, w)
        rcol.append(r)

def prepare_data(cutcol=['a', 'e'], datasets=None):
    print "prepare..."
    if datasets is None:
        datasets = map(rdb.loadObject, sources)
    data_arr = []
    for pha, dataset in zip([1,0,1,0], datasets):
        cutdata = dataset[cutcol]
        arr = cutdata.as_matrix()
        phacol = np.array([[pha]*len(arr)]).T
        arr_ = np.append(arr, phacol, axis=1)
        data_arr.append(arr_)
    return data_arr

def get_learndata(datasets, split=True):
    # datasets = prepare_data(cutcol)
    # print "gen..."
    join_train = np.concatenate((datasets[0], datasets[1]))
    join_test = np.concatenate((datasets[2], datasets[3]))

    if split:
        data_train = np.random.permutation(join_train)
        data_test = np.random.permutation(join_test)
        xdata_train, ydata_train = split_by_lastcol(data_train)
        xdata_test, ydata_test = split_by_lastcol(data_test)
        # print "xdata_train:", xdata_train[:5]
        return xdata_train, ydata_train, xdata_test, ydata_test
    else:
        mix = np.concatenate((join_train, join_test))
        permutate = np.random.permutation(mix)
        xdata, ydata = split_by_lastcol(permutate)
        return xdata, ydata

def update_datasets():
    loads = map(rdb.loadObject, sources)
    for dataset in loads:
        rdb.calc_rascend(dataset)
        rdb.calc_orbc(dataset)
        rdb.calc_rclose(dataset)
    [rdb.dumpObject(obj, fname) for obj, fname in zip(loads, sources)]



if __name__ == '__main__':
    print "init orbits update..."
    update_datasets()
    print "orbits update finished."



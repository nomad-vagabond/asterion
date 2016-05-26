import sys, os, trace, multiprocessing, time
from multiprocessing import Pool, Lock
import numpy as np
import pandas as pd
from numpy.linalg import norm
import pickle, urllib
from functools import partial
import calculate_orbits as co

database_url = "http://www.ianww.com/latest_fulldb.csv"
database_path = "./asteroid_data/latest_fulldb.csv"

print_msg = {
    'read_csv_start': "Loading asteroid database...",
    'read_csv_finish': "Asteroid database loaded in %f seconds.",
    'calc_moid_start': "init MOID copmutation...",
    'calc_moid_finish': "MOID copmutation finished in %f seconds.",
    'retrieve_database_start': "Downloading database...",
    'retrieve_database_finish': "Download finished in %f seconds."
    }


def print_jobtime(func):
    def wrapped(*args, **kwargs):
        func_name = func.__name__
        finish = False
        if kwargs.has_key('jobtime') and kwargs['jobtime']:
            print print_msg[func_name + '_start']
            t0 = time.time()
            finish = True
            del kwargs['jobtime']
        res = func(*args, **kwargs)
        if finish:
            t1 = time.time()
            print print_msg[func_name + '_finish'] % (t1-t0)
        return res
    return wrapped

# read_csv = print_jobtime(pd.read_csv)

# @print_jobtime
# def read_csv(*args, **kwargs):
#     return pd.read_csv(*args, **kwargs)

def load_database(columns, jobtime=False):
    if not os.path.exists(database_path):
        retrieve_database(database_url, database_path, jobtime=jobtime)
    read_csv = print_jobtime(pd.read_csv)
    database = read_csv(database_path, sep=',', usecols=columns, 
                        low_memory=False, jobtime=jobtime)
    return database

@print_jobtime
def retrieve_database(database_url, filepath):
    # if jobtime:
    #     print print_msg['download_start']
    # t0 = time.time()
    database = urllib.URLopener()
    database.retrieve(database_url, filepath)
    # t1 = time.time()
    # if jobtime:
    #     print print_msg['download_finish'] % (t1-t0)

def loadObject(fname):
    obj_file = open(fname,'r')
    obj = pickle.load(obj_file)
    obj_file.close()
    return obj

def dumpObject(obj, fname):
    obj_file = open(fname,'wb')
    pickle.dump(obj, obj_file)
    obj_file.close()

def cut_magnitude(database, target='under', threshold=22.0):
    if target == 'under':
        cut_h = database[database.H < threshold]
    else:
        cut_h = database[database.H >= threshold]
    return cut_h

def get_neo(database, subset):
    database_neo = database[database.neo == "Y"]
    # subset = ['pha', 'H', 'e', 'a', 'q', 'i', 'om', 'w', 'moid']
    database_clear = database_neo.dropna(subset=subset)
    return database_clear, len(database_clear)

def get_apollos(database):
    db1 = database[database.a > 1.0]
    db2 = db1[db1.q < 1.017]
    return db2, len(db2)

def get_atens(database):
    db1 = database[database.a < 1.0]
    db2 = db1[db1.q > 0.983]
    return db2, len(db2)

def get_amors(database):
    db1 = database[database.a > 1.0]
    db2 = db1[db1.q > 1.017]
    db3 = db2[db2.q < 1.3]
    return db3, len(db3)

def get_haz(database):
    haz = database[database['pha'] == 'Y']
    nohaz = database[database['pha'] == 'N']
    return haz, nohaz

def get_hazH(database):
    haz = database[database.H < 22.0]
    nohaz = database[database.H >= 22.0]
    return haz, nohaz

def get_hazMOID(database):
    haz = database[database.moid <= 0.05]
    nohaz = database[database.moid > 0.05]
    return haz, nohaz

def cutoff_outliers(data):
    data_cuta = data[data.a < 5.0]
    data_cuti = data_cuta[data_cuta.i < 100.]
    return data_cuti, len(data_cuti)

def append_moid(ir):
    index, row = ir
    w_, i_, om_ = np.radians([row.w, row.i, row.om])
    moid = co.get_moid(row.a, row.e, w_, i_, om_)
    # data.set_value(index, 'moid', moid)
    return (index, moid)

@print_jobtime
def calc_moid(data):
    """append column with values of moid"""
    corenums = multiprocessing.cpu_count()
    if corenums > 1:
        corenums -= 1
    pool = Pool(processes=corenums)
    joblist = [(index, row) for index, row in data.iterrows()]
    # pool.map(partial(append_moid, data=data), joblist)
    mapresult = pool.map_async(append_moid, joblist)
    pool.close()
    pool.join()
    for index, moid in mapresult.get():
        data.set_value(index, 'moid', moid)

# @print_jobtime
def calc_moid_1(data):
    for index, row in data.iterrows():
        w_, i_, om_ = np.radians([row.w, row.i, row.om])
        moid = co.get_moid(row.a, row.e, w_, i_, om_)
        data.set_value(index, 'moid', moid)

def calc_rascend(data):
    """append column with values of ascending node distance"""
    for index, row in data.iterrows():
        a, e, w_ = row.a, row.e, np.radians(row.w)
        r = co.get_r(a, e, w_)
        data.set_value(index, 'rasc', r)

def calc_rclose(data):
    """append columns with values of clocect distance
       in ecliptics plane and its z offset"""
    for index, row in data.iterrows():
        a, e = row.a, row.e
        w_, i_, om_ = np.radians([row.w, row.i, row.om])
        rx, ry = co.get_rxry(a, e, w_, i_, om_)
        data.set_value(index, 'rx', rx)
        data.set_value(index, 'ry', rx)

def calc_orbc(data):
    """append columns with values of orbit center coordinates"""
    for index, row in data.iterrows():
        a, e = row.a, row.e
        w_, i_, om_ = np.radians([row.w, row.i, row.om])
        c = co.find_center(a, e, w_, i_, om_)
        data.set_value(index, 'cx', c[0])
        data.set_value(index, 'cy', c[1])
        data.set_value(index, 'cz', c[2])


if __name__ == '__main__':

    ### READ ASTEROID DATABASE ###
    # reload(sys)
    # sys.path += ['.']
    database_path = "./asteroid_data/latest_fulldb.csv"
    if not os.path.exists(database_path):
        database_url = "http://www.ianww.com/latest_fulldb.csv"
        retrieve_database(database_url, database_path, jobtime=True)


    # folder_path = sys.path[0]
    # database_path = os.path.join(folder_path, os.path.dirname(""),
    #                 "./asteroid_data/latest_fulldb.csv")
    # database = pd.read_csv(database_path, sep=',', usecols=[0,4,6,7,8,9,15,
    #                        18,19,32,33,34,35,36,37,43,44,45,46,47,48,49])
    columns = ['a', 'e', 'i', 'w', 'om', 'q', 'H', 'neo', 
               'pha', 'per', 'n', 'ma', 'epoch', 'moid'] # 'moid'
    database = load_database(columns, jobtime=True)
    # database = read_csv(database_path, sep=',', usecols=columns,
    #                     low_memory=False, jobtime=True)

    # db_head = database[:10]
    print "db_head:\n", database.head(n=5)

    ### EXTRACT NEOS ###
    neo, num_neo = get_neo(database, columns)

    # ### RECALCULATE MOID BASED ON ORBITAL PARAMETERS ###
    # calc_moid(neo, jobtime=True)


    ### ADD ASCENDING NODE DISTANCE ###
    # calc_rascend(neo)
    print "neo:", neo[:10]

    ### SPLIT BY GROUPS ###
    apollos, num_apollos = get_apollos(neo)
    # print "apollos:", num_apollos, "neo:", num_neo, "ratio:", num_apollos/float(num_neo)

    atens, num_atens = get_atens(neo)
    # print "atens:", num_atens, "neo:", num_neo, "ratio:", num_atens/float(num_neo)

    amors, num_amors = get_amors(neo)
    # print "amors:", num_amors, "neo:", num_neo, "ratio:", num_amors/float(num_neo)

    # print "apollos:", apollos[:15]
    # print "amors:", amors[:15]

    ### CUT OFF OUTCASTS ###
    # apollos_cuti = apollos[apollos.i > 5.0]
    # apollos_cuti2 = apollos_cuti[apollos_cuti.i < 15.0]
    # apollos_cute = apollos[apollos.e < 0.5]
    # apollos_cuti2 = apollos_cuti[apollos_cuti.i < 15.0]
    apollos, num_apollos = cutoff_outliers(apollos)


    ### RECALCULATE MOID BASED ON ORBITAL PARAMETERS ###
    # calc_moid(apollos, jobtime=True)
    # neos, num_neos = cutoff_outliers(neo)

    ### REMOVE DIM ASTEROIDS ###
    # bright = cut_magnitude(apollos)
    # bright = cut_magnitude(apollos, target='above')

    ### SPLIT ASTEROIDS INTO BY PHA FLAG ###
    # haz, nohaz = get_hazMOID(apollos_cuti)
    # haz, nohaz = get_haz(bright)
    haz, nohaz = get_hazMOID(apollos)

    dumpObject(haz, './asteroid_data/haz_test.p')
    dumpObject(nohaz, './asteroid_data/nohaz_test.p')

    print "Load finished."
    print "number of hazardous asteroids in apollo group:", len(haz)
    print "number of nonhazardous asteroids in apollo group:", len(nohaz)






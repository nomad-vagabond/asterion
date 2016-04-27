import sys, os, trace, multiprocessing
import numpy as np
import pandas as pd
from numpy.linalg import norm
import pickle  

import calculate_orbits as co




def loadObject(fname):
    obj_file = open(fname,'r')
    obj = pickle.load(obj_file)
    obj_file.close()
    return obj

def dumpObject(obj, fname):
    obj_file = open(fname,'wb')
    pickle.dump(obj, obj_file)
    obj_file.close()

def cut_magnitude(database):
    cut_h = database[database.H < 22.0]
    return cut_h

def get_neo(database):
    database_neo = database[database.neo == "Y"]
    # print "database_neo sample:\n", database_neo[50:100]
    # print "len(database_neo):", len(database_neo)
    # database_clear = database_neo.dropna(subset=['pha', 'H', 'e', 'a', 'q',
    #                                                'i', 'om', 'w', 'per_y',
    #                                                'moid', 't_jup'])
    database_clear = database_neo.dropna(subset=['pha', 'H', 'e', 'a', 'q',
                                                 'i', 'om', 'w', 'moid'])
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

def cutoff_outcasts(data):
    data_cuta = data[data.a < 5.0]
    data_cuti = data_cuta[data_cuta.i < 100.]
    return data_cuti, len(data_cuti)

def calc_moid(data):
    """append column with values of moid"""
    for index, row in data.iterrows():
        w_, i_, om_ = np.radians([row.w, row.i, row.om])
        moid = co.get_moid(row.a, row.e, w_, i_, om_)
        data.set_value(index, 'moid', moid)
    return data

def calc_rascend(data):
    """append column with values of ascending node distance"""
    for index, row in data.iterrows():
        a, e, w_ = row.a, row.e, np.radians(row.w)
        r = co.get_r(a, e, w_)
        data.set_value(index, 'rasc', r)
    # return data

def calc_rclose(data):
    """append columns with values of clocect distance
       in ecliptics plane and its z offset"""
    for index, row in data.iterrows():
        a, e = row.a, row.e
        w_, i_, om_ = np.radians([row.w, row.i, row.om])
        rx, ry = co.get_rxry(a, e, w_, i_, om_)
        data.set_value(index, 'rx', rx)
        data.set_value(index, 'ry', rx)
    # return data

def calc_orbc(data):
    """append columns with values of orbit center coordinates"""
    for index, row in data.iterrows():
        a, e = row.a, row.e
        w_, i_, om_ = np.radians([row.w, row.i, row.om])
        c = co.find_center(a, e, w_, i_, om_)
        data.set_value(index, 'cx', c[0])
        data.set_value(index, 'cy', c[1])
        data.set_value(index, 'cz', c[2])
    # return data



if __name__ == '__main__':

    ### READ ASTEROID DATABASE ###
    folder_path = sys.path[0]
    database_path = os.path.join(folder_path, os.path.dirname(""),
                    "./asteroid_data/latest_fulldb.csv")
    # database = pd.read_csv(database_path, sep=',', 
    #            usecols=[0,4,6,7,8,9,15,18,19,32,33,34,35,36,37,43,44,45,46,47,48,49])
    database = pd.read_csv(database_path, sep=',', usecols=['a', 'e', 'i', 'w', 'om', 'q',
                                                            'H', 'neo', 'pha', 'moid', 'per',
                                                            'n', 'ma', 'epoch'],
                                                            low_memory=False)

    db_head = database[:10]
    print "db_head:\n", db_head

    ### EXTRACT NEOS ###
    neo, num_neo = get_neo(database)

    ### RECALCULATE MOID BASED ON ORBITAL PARAMETERS ###
    print "init MOID copmutation..."
    neo = calc_moid(neo)
    print "MOID copmutation finished."

    ### ADD ASCENDING NODE DISTANCE ###
    # calc_rascend(neo)
    # print "neo:", neo[:10]

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
    # apollos, num_apollos = cutoff_outcasts(apollos)

    neos, num_neos = cutoff_outcasts(neo)

    ### REMOVE DIM ASTEROIDS ###
    # bright = cut_magnitude(apollos)
    bright = cut_magnitude(neos)

    ### SPLIT ASTEROIDS INTO BY PHA FLAG ###
    # haz, nohaz = get_hazMOID(apollos_cuti)
    haz, nohaz = get_haz(bright)

    dumpObject(haz, './asteroid_data/haz_test.p')
    dumpObject(nohaz, './asteroid_data/nohaz_test.p')

    print "Load finished."
    print "number of hazardous asteroids in apollo group:", len(haz)
    print "number of nonhazardous asteroids in apollo group:", len(nohaz)






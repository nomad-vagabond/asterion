import numpy as np
import pickle 
import pandas as pd
# from read_database import calc_moid, get_hazMOID

import read_database as rdb
# from learn_data import loadObject, dumpObject

def gen_rand(data_arr, num=100):
    amin, amax = np.min(data_arr[:,0]), np.max(data_arr[:,0])
    emin, emax = np.min(data_arr[:,1]), np.max(data_arr[:,1])
    imin, imax = np.min(data_arr[:,2]), np.max(data_arr[:,2])
    wmin, wmax = np.min(data_arr[:,3]), np.max(data_arr[:,3])
    omin, omax = 0.0, 360.0
    qmin, qmax = np.min(data_arr[:,5]), np.max(data_arr[:,5])
    
    # ae = np.multiply(data_arr[:,0], data_arr[:,1])
    # aemin, aemax = np.min(ae), np.max(ae)
    arand = np.random.uniform(low=amin, high=amax, size=num)
    qrand = np.random.uniform(low=qmin, high=qmax, size=num)
    # aerand = np.random.uniform(low=aemin, high=aemax, size=num)
    erand = np.array([(arand[i]-qrand[i])/arand[i] for i in range(int(num))])

    randdata = np.array([arand,
                         erand,
                         np.random.uniform(low=imin, high=imax, size=num),
                         np.random.uniform(low=wmin, high=wmax, size=num),
                         np.random.uniform(low=omin, high=omax, size=num),
                         qrand]).T
    return randdata

if __name__ == '__main__':

    haz = rdb.loadObject('./asteroid_data/haz.p')
    nohaz = rdb.loadObject('./asteroid_data/nohaz.p')

    haz_data = haz[['a', 'e', 'i', 'w', 'om', 'q']]
    nohaz_data = nohaz[['a', 'e', 'i', 'w', 'om', 'q']]
    hazdata_arr = haz_data.as_matrix()
    nohazdata_arr = nohaz_data.as_matrix()

    data_full = np.concatenate((hazdata_arr, nohazdata_arr))

    print "init orbit generation..."
    randdata = gen_rand(data_full, num=1e2)
    print "orbit generation finished."

    dataframe = pd.DataFrame(randdata, columns=['a', 'e', 'i', 'w', 'om', 'q'])
    ### CALCULATE MOID ###
    print "init MOID copmutation..."
    data = rdb.calc_moid(dataframe)
    print "MOID copmutation finished."

    # print "randdata sample:", randdata[:10]
    print "dataframe:\n", dataframe[:10]

    haz, nohaz = rdb.get_hazMOID(data)

    rdb.dumpObject(haz, './asteroid_data/haz_rand_test.p')
    rdb.dumpObject(nohaz, './asteroid_data/nohaz_rand_test.p')


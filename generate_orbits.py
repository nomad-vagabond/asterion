import numpy as np
import pickle 
import pandas as pd
import read_database as rdb
from math import pi

G = 6.67384e-11
M = 1.989e30

def get_param_bounds(haz, nohaz, names):
    data_full = pd.concat([haz[names], nohaz[names]])
    maxvals = [np.max(data_full[name]) for name in names]
    minvals = [np.min(data_full[name]) for name in names]
    params = ({name:(minval, maxval) 
              for name, minval, maxval in zip(names, minvals, maxvals)})
    # print "params.items():", params.items()
    return params

def gen_rand_params(params=None, num=1):
    if params is None:
        params = rdb.loadObject('./asteroid_data/orbparams_minmax.p')
    rand_params = ({name:np.random.uniform(low=values[0], high=values[1], 
                    size=num) for name, values in params.items()})
    rand_params['e'] = (rand_params['a'] - rand_params['q'])/rand_params['a']
    rand_params['per'] = 2*pi*np.sqrt(rand_params['a']**3/(G*M))/86400.0
    # if num == 1:
    #     print "rand_params:", rand_params
    return rand_params

def gen_rand_orbits(params, names, num=100):
    rand_params = gen_rand_params(params=params, num=num)
    randdata = np.array([rand_params[name] for name in names]).T
    return randdata


if __name__ == '__main__':

    haz = rdb.loadObject('./asteroid_data/haz_test.p')
    nohaz = rdb.loadObject('./asteroid_data/nohaz_test.p')

    names = ['a', 'i', 'w', 'om', 'q', 'n', 'ma', 'epoch']
    params = get_param_bounds(haz, nohaz, names)
    rdb.dumpObject(params, './asteroid_data/orbparams_minmax.p')
    gen_rand_params(params)
    
    print "init orbit generation..."
    names = ['a', 'e', 'i', 'w', 'om', 'q']
    randdata = gen_rand_orbits(params, names, num=2e3)
    print "orbit generation finished.", randdata.shape

    dataframe = pd.DataFrame(randdata, columns=names)
    
    ### CALCULATE MOID ###
    print "init MOID copmutation..."
    data = rdb.calc_moid(dataframe)
    print "MOID copmutation finished."

    # print "randdata sample:", randdata[:10]
    print "dataframe:\n", dataframe[:10]

    haz, nohaz = rdb.get_hazMOID(data)

    rdb.dumpObject(haz, './asteroid_data/haz_rand_test.p')
    rdb.dumpObject(nohaz, './asteroid_data/nohaz_rand_test.p')


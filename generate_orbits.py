import numpy as np
import pickle 
import pandas as pd
# from read_database import calc_moid, get_hazMOID
import scipy.stats as ss
import read_database as rdb
import matplotlib.pyplot as plt
import scipy
# from learn_data import loadObject, dumpObject



def get_density(data, num=30):
    dmin, dmax = min(data), max(data)
    # datanorm = (data - dmin)/(dmax - dmin)
    datanorm = data
    size = len(data)
    print "size:", size
    print "dmin, dmax:", dmin, dmax
    bounds = np.linspace(dmin, dmax, num)
    # bounds = np.linspace(0, 1, num)
    x = scipy.arange(size)
    x = np.linspace(dmin, dmax, num*16)
    # bounds = np.concatenate(([0.0], bounds))
    # bounds_centers = np.array([(a+b)*0.5 for a, b in zip(bounds[:-1], bounds[1:])])
    # density = np.histogram(data - dmin, bins=bounds, density=True)[0]
    # widths = np.array([(b - a) for a, b in zip(bounds[:-1], bounds[1:])])

    h = plt.hist(datanorm, bins=bounds, color='w')


    # pp = np.linspace(dmin, dmax, num*16)
    pp = np.linspace(0, 1, num*16)
    # print "density:", density, len(density)
    print "bounds:", bounds, len(bounds)
    # h = plt.hist(density, bins=range(num), color='y')
    gamma_pdf = ss.gamma
    a, loc, scale = gamma_pdf.fit(datanorm) # scale=dmax-dmin
    print "a:", a
    print "loc:", loc
    print "scale:", scale
    # pdf_fitted = gamma_pdf.pdf(bounds, *param[:-2], loc=param[-2], scale=param[-1])
    # bb = np.linspace(dmin, dmax, num*3)
    # pdf_fitted = gamma_pdf.pdf(bounds, 1.99)
    pdf_fitted = gamma_pdf.pdf(x, a, loc=loc, scale=scale)
    # plt.bar(bounds_centers, density, widths[0], color='w')
    # h = plt.hist(density, bounds_centers, color='w')
    # pp = np.linspace(0, dmax, num*4)
    # l = plt.plot(bounds, pdf_fitted, 'r--', linewidth=1)
    # l = plt.plot(pp, gamma_pdf.pdf(pp, 100, scale=scale), 'r--', linewidth=1)
    l = plt.plot(x, pdf_fitted, 'r--', linewidth=1)
    plt.xlim(dmin, dmax)
    plt.show()




def get_density0(data, num=20):
    dmin, dmax = min(data), max(data)
    size = len(data)
    print "dmin, dmax:", dmin, dmax
    bounds = np.linspace(0, dmax-dmin, num)
    # bounds = np.concatenate(([0.0], bounds))
    bounds_centers = np.array([(a+b)*0.5 for a, b in zip(bounds[:-1], bounds[1:])])
    density = np.histogram(data - dmin, bins=bounds, density=True)[0]
    widths = np.array([(b - a) for a, b in zip(bounds[:-1], bounds[1:])])

    print "widths:", widths
    pdf_sum = sum(d*w for d, w in zip(density, widths))
    print "pdf_sum:", pdf_sum



    pp = np.linspace(0, dmax-dmin, num*8)
    print "density:", density, len(density)
    print "bounds:", bounds, len(bounds)
    # h = plt.hist(density, bins=range(num), color='y')
    gamma_pdf = ss.gamma
    a, loc, scale = gamma_pdf.fit(density, scale=dmax-dmin) # scale=dmax-dmin
    print "a:", a
    print "loc:", loc
    print "scale:", scale
    # pdf_fitted = gamma_pdf.pdf(bounds, *param[:-2], loc=param[-2], scale=param[-1])
    # bb = np.linspace(dmin, dmax, num*3)
    # pdf_fitted = gamma_pdf.pdf(bounds, 1.99)
    pdf_fitted = gamma_pdf.pdf(pp, a, loc=loc, scale=scale)
    # print "pdf_fitted:", pdf_fitted

    # param = gamma_pdf.fit(density)
    # print "param:", param
    # pdf_fitted = gamma_pdf.pdf(pp, *param[:-2], loc=param[-2], scale=param[-1])  # *(dmax-dmin)


    # n, bins, patches = plt.hist(density, bins=range(len(density))) # bins=bounds
    # gaussian_numbers = np.random.randn(1000)
    # print "gaussian_numbers:", gaussian_numbers
    # plt.hist(gaussian_numbers, bins=bounds)
    plt.bar(bounds_centers, density, widths[0], color='w')
    # h = plt.hist(density, bounds_centers, color='w')
    # pp = np.linspace(0, dmax, num*4)
    # l = plt.plot(bounds, pdf_fitted, 'r--', linewidth=1)
    # l = plt.plot(pp, gamma_pdf.pdf(pp, 100, scale=scale), 'r--', linewidth=1)
    l = plt.plot(pp, pdf_fitted, 'r--', linewidth=1)
    plt.show()




def gen_rand_asteroid(data_arr):
    params = get_param_bounds(data_arr)
    # print "params a:", params['a'][0], params['a'][1]
    a = np.random.uniform(low=params['a'][0], high=params['a'][1], size=1)[0]
    i = np.random.uniform(low=params['i'][0], high=params['i'][1], size=1)[0]
    w = np.random.uniform(low=params['w'][0], high=params['w'][1], size=1)[0]
    om = np.random.uniform(low=params['om'][0], high=params['om'][1], size=1)[0]
    q = np.random.uniform(low=params['q'][0], high=params['q'][1], size=1)[0]
    e = (a - q)/a
    return a, e, w, i, om





def get_param_bounds(data_arr):
    amin, amax = np.min(data_arr[:,0]), np.max(data_arr[:,0])
    # emin, emax = np.min(data_arr[:,1]), np.max(data_arr[:,1])
    imin, imax = np.min(data_arr[:,2]), np.max(data_arr[:,2])
    wmin, wmax = np.min(data_arr[:,3]), np.max(data_arr[:,3])
    omin, omax = 0.0, 360.0
    qmin, qmax = np.min(data_arr[:,5]), np.max(data_arr[:,5])
    params = {'a': (amin, amax), 
              'i': (imin, imax), 
              'w': (wmin, wmax), 
              'om': (omin, omax), 
              'q': (qmin, qmax)}
    return params





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





    get_density(hazdata_arr[:,0])

    # print "init orbit generation..."
    # randdata = gen_rand(data_full, num=1e2)
    # print "orbit generation finished."

    # dataframe = pd.DataFrame(randdata, columns=['a', 'e', 'i', 'w', 'om', 'q'])
    # ### CALCULATE MOID ###
    # print "init MOID copmutation..."
    # data = rdb.calc_moid(dataframe)
    # print "MOID copmutation finished."

    # # print "randdata sample:", randdata[:10]
    # print "dataframe:\n", dataframe[:10]

    # haz, nohaz = rdb.get_hazMOID(data)

    # rdb.dumpObject(haz, './asteroid_data/haz_rand_test.p')
    # rdb.dumpObject(nohaz, './asteroid_data/nohaz_rand_test.p')


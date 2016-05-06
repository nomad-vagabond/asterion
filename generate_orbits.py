import numpy as np
import pickle 
import pandas as pd
import read_database as rdb
from math import pi, sqrt
# from read_database import calc_moid, get_hazMOID
import scipy.stats as ss
import scipy.optimize as so
import read_database as rdb
import matplotlib.pyplot as plt
import scipy
# from learn_data import loadObject, dumpObject

G = 6.67384e-11
M = 1.989e30
AU = 149597870700

class ContinuousDistribution(object):
    """
    Fitted continuaous distribution.
    Fits data with continuous distribution specified as distfunc.

    Parameters
    ----------
    data: 1-D array of independent imput data.

    distfunc: scipy.stats countinuous random variable class.
              Currently supports continuous random variables with 
              shape parameter.
    """
    def __init__(self, data, distfunc):
        self.distfunc = distfunc
        self.dmin, self.dmax = min(data), max(data)
        self.get_histogram(data)
        self.get_distribution(data)
        # return self.distfit

    def get_histogram(self, data, num=50):
        bounds = np.linspace(0, self.dmax, num)
        sections = zip(bounds[:-1], bounds[1:])
        self.probs = np.histogram(data, bins=bounds, density=True)[0]
        self.sections_c = np.array([(a+b)*0.5 for a, b in sections])
        self.widths = np.array([(b - a) for a, b in sections])
        self.bounds = bounds[:-1]
        self._check_histogram()
        # print "widths:", widths

    def get_distribution(self, data, num=50):
        pdf = self.distfunc.pdf
        distshapes = self.distfunc.shapes
        if distshapes is None:
            f = lambda x, scale: pdf(x, loc=self.dmin, scale=scale)
            self.distfit = self.distfunc(scale=self.dmax)
            cdf = self.distfunc.cdf(self.dmax)
        elif len(distshapes.split()) == 1:
            f = lambda x, shape, scale: pdf(x, shape, loc=self.dmin, scale=scale)
            popt, pcov = so.curve_fit(f, self.sections_c, self.probs, p0=[1.2, 1])
            # print "popt:", popt
            shape = popt[0]
            scale = popt[1]
            self.distfit = self.distfunc(shape, loc=self.dmin, scale=scale)
            cdf = self.distfunc.cdf(self.dmax, shape, loc=self.dmin, scale=scale)
        elif len(distshapes.split()) == 2:
            f = lambda x, shape1, shape2, scale: pdf(x, shape1, shape2, loc=self.dmin, scale=scale)
            popt, pcov = so.curve_fit(f, self.sections_c, self.probs, p0=[1.2, 1.2, 1]) 
            # print "popt:", popt
            shapes = popt[:2]
            scale = popt[2]
            self.distfit = self.distfunc(shapes[0], shapes[1], loc=self.dmin, scale=scale)
            cdf = self.distfunc.cdf(self.dmax, shapes[0], shapes[1], loc=self.dmin, scale=scale)
        else:
            raise ValueError("specified distribution currently is not supported")
        print "cdf(dmax):", cdf

    def _check_histogram(self):
        pdf_sum = sum(d*w for d, w in zip(self.probs, self.widths))
        print "pdf_sum:", pdf_sum

    def get_rvs(self, size=100):
        rvs = self.distfit.rvs(size=size)
        # print "rvs:", rvs
        return rvs

    def plot_distfit(self, npoints=100):
        ppx = np.linspace(0, self.dmax, npoints)
        ppy = self.distfit.pdf(ppx)
        plt.bar(self.sections_c, self.probs, self.widths[0], color='w', alpha=0.7)
        plt.plot(ppx, ppy, 'r-', lw=2)
        plt.show()
        # pass

    def plot_rvs(self, npoints=1000):
        rvs = self.get_rvs(size=npoints)
        bounds = np.linspace(0, self.dmax, 50)
        plt.hist(rvs, bins=bounds, normed=1, color='grey')
        self.plot_distfit()


def get_subplotnum(n):
    b = int(sqrt(n))
    a = int(float(n)/b)
    m = n % b
    if m > 0:
        a +=1
    return str(a) + str(b)

def plot_param_distributions(distlist, xlabels, npoints=1000):
    fig = plt.figure()
    subplot_base = get_subplotnum(len(distlist))
    subplots = [int(subplot_base + str(i+1)) for i in range(len(distlist))]
    for subplot, dist, xlabel in zip(subplots, distlist, xlabels):
        rvs = dist.get_rvs(size=npoints)
        ax = fig.add_subplot(subplot)
        ax.grid(True)
        bounds = np.linspace(0, dist.dmax, 50)
        ax.hist(rvs, bins=bounds, normed=1, color='slategray', lw=0, zorder=0)
        ppx = np.linspace(0, dist.dmax, npoints)
        ppy = dist.distfit.pdf(ppx)
        ax.bar(dist.bounds, dist.probs, dist.widths[0], color='lightsteelblue', alpha=0.6, zorder=1)
        distcolor = 'cornflowerblue' # 'limegreen'
        ax.plot(ppx, ppy, color=distcolor, ls='-', lw=2, zorder=3)
        ax.fill_between(ppx, 0, ppy, facecolor=distcolor, zorder=2, alpha=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, None)
        ax.set_xlim(0, dist.dmax)
    plt.show()

def get_param_distributions(data, names, statdists):
    contdists = [ContinuousDistribution(data[[name]].as_matrix().ravel(), dist)
                 for name, dist in zip(names, statdists)]
    return contdists

def get_param_bounds(haz, nohaz, names):
    data_full = pd.concat([haz[names], nohaz[names]])
    maxvals = [np.max(data_full[name]) for name in names]
    minvals = [np.min(data_full[name]) for name in names]
    params = ({name:(minval, maxval) 
              for name, minval, maxval in zip(names, minvals, maxvals)})
    # print "params.items():", params.items()
    return params

def gen_rand_params(params=None, distdict=None, num=1):
    if distdict is None:
        if params is None:
            params = rdb.loadObject('./asteroid_data/orbparams_minmax.p')
        rand_params = ({name:np.random.uniform(low=values[0], high=values[1], 
                        size=num) for name, values in params.items()})
    else:
        rand_params = ({name: contdist.get_rvs(size=num)
                        for name, contdist in distdict.items()})
    try:
        rand_params['e'] = (rand_params['a'] - rand_params['q'])/rand_params['a']
        rand_params['per'] = 2*pi*np.sqrt((rand_params['a']*AU)**3/(G*M))/86400.0
    except:
        pass
    # if num == 1:
    #     print "rand_params:", rand_params
    return rand_params

def gen_rand_orbits(params, names, distlist, num=100):
    distdict = {name:dist for name, dist in zip(names, distlist)}
    rand_params = gen_rand_params(distdict=distdict, num=num)
    names_extend = rand_params.keys()
    randdata = np.array([rand_params[name] for name in names_extend]).T
    dataframe = pd.DataFrame(randdata, columns=names_extend)
    return dataframe



if __name__ == '__main__':

    haz = rdb.loadObject('./asteroid_data/haz_test.p')
    nohaz = rdb.loadObject('./asteroid_data/nohaz_test.p')

    names = ['a', 'i', 'w', 'om', 'q', 'n', 'ma', 'epoch']
    params = get_param_bounds(haz, nohaz, names)
    rdb.dumpObject(params, './asteroid_data/orbparams_minmax.p')
    gen_rand_params(params=params)
    
    print "init orbit generation..."
    # names = ['a', 'e', 'i', 'w', 'om', 'q']
    names = ['a', 'i', 'w', 'om', 'q']
    statdists = [ss.chi, ss.gamma, ss.uniform, ss.uniform, ss.beta, ss.uniform, ss.beta]
    data_full = pd.concat([haz[names], nohaz[names]])
    distlist = get_param_distributions(data_full, names, statdists)
    randdata = gen_rand_orbits(params, names, distlist, num=100)
    print "orbit generation finished."
    print "randdata sample:\n", randdata[:5]
    plot_param_distributions(distlist, names)
    
    ### CALCULATE MOID ###
    print "init MOID copmutation..."
    data = rdb.calc_moid(randdata)
    print "MOID copmutation finished."
    # haz, nohaz = rdb.get_hazMOID(data)

    ### DUMP RANDOM ORBITS ###
    rdb.dumpObject(haz, './asteroid_data/haz_rand_test.p')
    rdb.dumpObject(nohaz, './asteroid_data/nohaz_rand_test.p')





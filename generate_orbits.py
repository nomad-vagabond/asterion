import pickle
import time
import string
from math import pi, sqrt, sin, copysign
from functools import partial

import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy.optimize as so
import scipy.integrate as si
# from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
# import scipy

import read_database as rdb

### ======================= ###

G = 6.67384e-11
M = 1.989e30
AU = 149597870700


class GaussianKDE(object):
    """1D Wrapper over scipy's gaussian_kde"""
    def __init__(self, data, name='gaussian_kde'):
        if type(data) == type(pd.DataFrame()):
            data_ = data.as_matrix()
        else:
            data_ = data
        self.dmin, self.dmax = min(data_), max(data_)
        self.name = name
        self.gaussian_kde = ss.gaussian_kde(data_)
        self.shapes = None

    def __call__(self, *args, **kwargs):
        return self

    def pdf(self, x, *args, **kwargs):
        return self.gaussian_kde.pdf(x)

    def cdf(self, *args, **kwargs):
        return self.gaussian_kde.integrate_box_1d(self.dmin, self.dmax)

    def rvs(self, size=None):
        rvs = self.gaussian_kde.resample(size=size)
        # print "rvs:", rvs[:5], rvs.shape
        return rvs.ravel()

        
class HarmonicDistribution(object):
    """ 1D harmonic continuous distribution."""
    def __init__(self, dmin=0, dmax=360):
        self.dmin, self.dmax = dmin, dmax
        self.name = 'harmonic'
        self.shapes = 'amp'
        self.frozen = False

    def __call__(self, amp, *args, **kwargs):
        self.amp = amp
        self.scale = kwargs['scale']
        self.loc = kwargs['loc']
        self.frozen = True
        return self

    def _pdf(self, x, amp, pha, shift):
        y = amp*np.sin(2*np.radians(x) + pha) + shift
        return y

    def pdf(self, x, amp=None, *args, **kwargs):
        if not self.frozen:
            amp_ = amp
            loc, scale = kwargs['loc'], kwargs['scale']
        else:
            amp_ = self.amp
            loc, scale = self.loc, self.scale
        return self._pdf(x, amp_, loc, scale)

    def cdf(self, x, *args, **kwargs):
        cdf_ = si.quad(self._pdf, 0, x, args=(self.amp, self.loc, self.scale))
        # print "cdf_:", cdf_
        return cdf_[0]

    def rvs(self, size=None, resolution=30):
        size_cut = int(size*0.02)
        x = np.linspace(self.dmin, self.dmax, resolution)
        w = x[1] - x[0]
        p0 = self._pdf(x[:-1] + w*0.5, self.amp, self.loc, self.scale)*w
        p = np.asarray(np.round(p0*(size-size_cut)),  dtype=int)
        # print "p_sum:", np.sum(p)
        sections = zip(x[:-1], x[1:], p)
        rvs_base = np.asarray([np.random.uniform(low=a, high=b, size=n) 
                               for a, b, n in sections])
        rvs_add = np.random.uniform(low=self.dmin, high=self.dmax, 
                                    size=(size - np.sum(p)))
        rvs = np.hstack(np.concatenate((rvs_base, rvs_add)))
        # return np.random.uniform(low=0, high=360, size=size)
        return rvs


class BimodalDistribution(object):
    """ 1D bimodal continuous distribution."""
    def __init__(self, dist1=ss.norm, dist2=ss.norm, magnitude=0.5, name='bimodal'):
        self.dist1 = dist1
        self.dist2 = dist2
        self.magnitude = magnitude
        self.name = name
        self.shapes = 'offset'
        self.frozen = False

    def __call__(self, offset, *args, **kwargs):
        self.offset = offset
        self.scale = kwargs['scale']
        self.loc = kwargs['loc']
        self.frozen = True
        return self

    def _parse_args(self, offset, *args, **kwargs):
        if not self.frozen:
            offset_, loc, scale = offset, kwargs['loc'], kwargs['scale']
        else:
            offset_, loc, scale = self.offset, self.loc, self.scale
        return offset_, loc, scale

    def _pdf(self, x, offset, loc, scale):
        pdf1 = self.dist1.pdf(x, loc=loc, scale=scale)
        pdf2 = self.dist2.pdf(x, loc=offset, scale=scale)
        bimodal_pdf = self.magnitude*pdf1 + (1-self.magnitude)*pdf2
        return bimodal_pdf

    def pdf(self, x, offset=180, *args, **kwargs):
        offset_, loc, scale = self._parse_args(offset, *args, **kwargs)
        return self._pdf(x, offset_, loc, scale)

    def cdf(self, x, offset=180, *args, **kwargs):
        offset_, loc, scale = self._parse_args(offset, *args, **kwargs)
        cdf_ = si.quad(self._pdf, 0, x, args=(offset_, loc, scale))
        return cdf_[0]
    
    def rvs(self, size=50):
        dist1 = self.dist1(loc=self.loc, scale=self.scale)
        dist2 = self.dist2(loc=self.offset, scale=self.scale)
        rvs1 = dist1.rvs(size=size*self.magnitude)
        rvs2 = dist2.rvs(size=size*(1-self.magnitude))
        rvs = np.hstack([rvs1, rvs2])
        # rvs = np.concatenate((rvs1, rvs2))
        return rvs


class FitDist(object):
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
    def __init__(self, data, distfunc, n=50, verbose=False):
        self.distfunc = distfunc
        self.dmin, self.dmax = min(data), max(data)
        pdf_sum = self._split(data, n)
        cdf_max = self._fit()
        if verbose:
            print "Data cdf(xmax): %f \t" % pdf_sum, 
            print "%s_cdf(xmax): %f" % (distfunc.name, cdf_max) 

    def _split(self, data, num):
        """Split data values into bands"""
        bounds = np.linspace(0, self.dmax, num)
        sections = zip(bounds[:-1], bounds[1:])
        self.probs = np.histogram(data, bins=bounds, density=True)[0]
        self.sections_c = np.array([(a+b)*0.5 for a, b in sections])
        self.widths = np.array([(b - a) for a, b in sections])
        # self.bounds = bounds[:-1]
        self.bounds = bounds
        pdf_sum = sum(d*w for d, w in zip(self.probs, self.widths))
        return pdf_sum

    def _fgen(self, shapes, pdf):
        """Generate function for curve fitting"""
        if shapes is None:
            shapes = ''
        else:
            shapes += ','
        # shapes = string.join('shape%d, ' %d for d in range(n_shapes))
        fdef = ("f = lambda x, %sloc, scale:"
             "pdf(x, %sloc=loc, scale=scale)" % (shapes, shapes))
        exec fdef in locals()
        return f

    def _fit(self):
        """Fit value bands with continuous distribution"""
        pdf = self.distfunc.pdf
        distshapes = self.distfunc.shapes

        # if distshapes is None:
        #     f = lambda x, loc, scale: pdf(x, loc=loc, scale=scale)
        # else:

        # n = len(distshapes.split())
        f = self._fgen(distshapes, pdf)

        if self.distfunc.name == 'uniform':
            self.distfit = self.distfunc(loc=self.dmin, scale=self.dmax)
            cdf = self.distfunc.cdf(self.dmax)
        else:
            popt, pcov = so.curve_fit(f, self.sections_c, self.probs)
            shapes = popt[:-2]
            self.distfit = self.distfunc(*shapes, loc=popt[-2], scale=popt[-1])
            cdf = self.distfunc.cdf(self.dmax, *shapes, 
                                    loc=popt[-2], scale=popt[-1])

        return cdf

    def get_rvs(self, size=100):
        """Returns random variables using fitted continuous distribution"""
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

def plot_param_distributions(distlist, xlabels, npoints=1000, figsize=(16, 10)):
    fig = plt.figure(figsize=figsize)
    subplot_base = get_subplotnum(len(distlist))
    subplots = [int(subplot_base + str(i+1)) for i in range(len(distlist))]
    for subplot, dist, xlabel in zip(subplots, distlist, xlabels):
        rvs = dist.get_rvs(size=npoints)
        ax = fig.add_subplot(subplot)
        # ax.grid(True)
        w = dist.widths[0]
        # bounds = dist.bounds - w*0.25
        ax.hist(rvs, bins=dist.bounds-w*0.25, normed=1, rwidth=0.5, 
                color='lightsteelblue', lw=0, zorder=1) # 'aquamarine'  'lightblue'
        ppx = np.linspace(0, dist.dmax, npoints)
        ppy = dist.distfit.pdf(ppx)
        ax.bar(dist.bounds[:-1]+w*0.5, dist.probs, w*0.5, lw=0, 
               color='cornflowerblue', alpha=1, zorder=2) # 'dodgerblue'
        distcolor = 'chocolate' # 'greenyellow' # 'limegreen' # 'cornflowerblue'
        ax.plot(ppx, ppy, color=distcolor, ls='--', lw=2, zorder=3)
        ax.fill_between(ppx, 0, ppy, facecolor=distcolor, zorder=0, alpha=0.1)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, None)
        ax.set_xlim(0, dist.dmax)
    plt.show()

def get_param_distributions(data, names, statdists, n=50, verbose=False):
    contdists = [FitDist(data[[name]].as_matrix().ravel(), 
                 dist, n=n, verbose=verbose) for name, dist in zip(names, statdists)]
    return contdists

def get_param_bounds(data, names):
    # data_full = pd.concat([haz[names], nohaz[names]])
    maxvals = [np.max(data[name]) for name in names]
    minvals = [np.min(data[name]) for name in names]
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
    data_full = pd.concat([haz[names], nohaz[names]])
    params = get_param_bounds(data_full, names)
    rdb.dumpObject(params, './asteroid_data/orbparams_minmax.p')
    gen_rand_params(params=params)
    
    print "init orbit generation..."
    # names = ['a', 'e', 'i', 'w', 'om', 'q']
    # gkde = GaussianKDE('gkde', data_full['w'].as_matrix())
    # gkde2 = GaussianKDE('gkde2', data_full['om'].as_matrix())
    # gkde_a = GaussianKDE('gkde_a', data_full['a'].as_matrix())
    kde_a = GaussianKDE(data_full['a'])
    names = ['a', 'i', 'w', 'om', 'q']
    bimod = BimodalDistribution()  # ss.logistic, ss.logistic
    statdists = [ss.johnsonsb, ss.exponweib, HarmonicDistribution(), HarmonicDistribution(), ss.genlogistic] # ss.exponweib ss.loggamma
    # ss.genlogistic ss.exponweib ss.loggamma ss.burr
    # ss.fatiguelife ss.foldnorm ss.genpareto ss.gompertz!!!  ss.johnsonsb!!! ss.pearson3 ss.powerlognorm ss.recipinvgauss
    # ss.uniform, ss.beta
    data_full = pd.concat([haz[names], nohaz[names]])
    distlist = get_param_distributions(data_full, names, statdists, n=30, verbose=True)
    randdata = gen_rand_orbits(params, names, distlist, num=1e5)
    print "orbit generation finished."
    print "randdata sample:\n", randdata[:5]
    plot_param_distributions(distlist, names)
    
    ### CALCULATE MOID ###
    # print "init MOID copmutation..."
    # t0 = time.time()
    data = rdb.calc_moid(randdata)
    # t1 = time.time() - t0
    # print "MOID copmutation finished in %f seconds." % t1
    # haz, nohaz = rdb.get_hazMOID(data)

    ### DUMP RANDOM ORBITS ###
    haz_rand, nohaz_rand = rdb.get_hazMOID(randdata)
    rdb.dumpObject(haz_rand, './asteroid_data/haz_rand_test.p')
    rdb.dumpObject(nohaz_rand, './asteroid_data/nohaz_rand_test.p')



dist_names = ['alpha', 
              'anglit', 
              'arcsine', 
              'beta', 
              'betaprime', 
              'bradford', 
              'burr', 
              'cauchy', 
              'chi', 
              'chi2', 
              'cosine', 
              'dgamma', 
              'dweibull', 
              'erlang', 
              'expon', 
              'exponweib', 
              'exponpow', 
              'f', 
              'fatiguelife', 
              'fisk', 
              'foldcauchy', 
              'foldnorm', 
              'frechet_r', 
              'frechet_l', 
              'genlogistic', 
              'genpareto', 
              'genexpon', 
              'genextreme', 
              'gausshyper', 
              'gamma', 
              'gengamma', 
              'genhalflogistic', 
              'gilbrat', 
              'gompertz', 
              'gumbel_r', 
              'gumbel_l', 
              'halfcauchy', 
              'halflogistic', 
              'halfnorm', 
              'hypsecant', 
              'invgamma', 
              'invgauss', 
              'invweibull', 
              'johnsonsb', 
              'johnsonsu', 
              'ksone', 
              'kstwobign', 
              'laplace', 
              'logistic', 
              'loggamma', 
              'loglaplace', 
              'lognorm', 
              'lomax', 
              'maxwell', 
              'mielke', 
              'nakagami', 
              'ncx2', 
              'ncf', 
              'nct', 
              'norm', 
              'pareto', 
              'pearson3', 
              'powerlaw', 
              'powerlognorm', 
              'powernorm', 
              'rdist', 
              'reciprocal', 
              'rayleigh', 
              'rice', 
              'recipinvgauss', 
              'semicircular', 
              't', 
              'triang', 
              'truncexpon', 
              'truncnorm', 
              'tukeylambda', 
              'uniform', 
              'vonmises', 
              'wald', 
              'weibull_min', 
              'weibull_max', 
              'wrapcauchy'] 


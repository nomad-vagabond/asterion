import pickle
import time
import string
# import os
from math import pi, sqrt, sin, copysign, floor, ceil
from functools import partial
import warnings

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

msg_ehigh = 'too high eccentricity is found. value has been reset to 0.99'



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

    def rvs(self, size=None, resolution=60):
        size = int(size)
        if size < 4:
            rvs = np.random.uniform(low=self.dmin, high=self.dmax, size=size)
            return rvs
        if size < resolution:
            resolution = int(ceil(size*0.33))
            # print "resolution:", resolution
        
        x = np.linspace(self.dmin, self.dmax, resolution)
        w = x[1] - x[0]
        p0 = self._pdf(x[:-1] + w*0.5, self.amp, self.loc, self.scale)*w
        # size_cut = max(int(size*0.02), 1)
        size_cut = 0
        for iteration in range(size):
            p = np.asarray(np.round(p0*(size-size_cut)),  dtype=int)
            psum = np.sum(p)
            if psum <= size:
                break
            else:
               size_cut += 1 
        # psum = min(np.sum(p), size)
        # print "p_sum:", np.sum(p)
        # print "size_cut:", size_cut
        # print "np.sum(p):", np.sum(p)
        sections = zip(x[:-1], x[1:], p)
        # print "sections:", sections
        rvs_base = np.asarray([np.random.uniform(low=a, high=b, size=n) 
                               for a, b, n in sections])
        rvs_add = np.random.uniform(low=self.dmin, high=self.dmax, 
                                    size=(size - psum))
        # rvs_base = rvs_base.ravel()
        rvs_base = np.hstack(rvs_base)
        # print "rvs_base:", rvs_base, rvs_base.shape
        # print "rvs_add:", rvs_add, rvs_add.shape
        rvs = np.hstack(np.concatenate((rvs_base, rvs_add)))
        # rvs = np.random.permutation(rvs_)
        
        # print "len(rvs):", len(rvs)
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
    # __module__ = os.path.splitext(os.path.basename(__file__))[0]
    
    def __init__(self, data, distfunc, n=50, verbose=False):
        self.distfunc = distfunc
        self.dmin, self.dmax = min(data), max(data)
        n_ = self._extend(n)
        pdf_sum = self._split(data, n_)
        cdf_max = self._fit()
        if verbose:
            print "Data cdf(xmax): %f \t" % pdf_sum, 
            print "%s_cdf(xmax): %f" % (distfunc.name, cdf_max) 

    def _extend(self, n):
        n_ = float(self.dmax) * n/(self.dmax - self.dmin)
        return int(n_)


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
            # cdf = self.distfunc.cdf(self.dmax, *shapes, 
            #                         loc=popt[-2], scale=popt[-1])
            cdf = si.quad(self.distfit.pdf, self.dmin, self.dmax)[0]

        return cdf


    def _cut_tails(self, rvs):

        below_bounds = np.where(rvs < self.dmin)[0]
        # print "below_bounds:", below_bounds.shape, #type(below_bounds)
        above_bounds = np.where(rvs > self.dmax)[0]
        # print "above_bounds:", above_bounds.shape, #type(below_bounds)
        bad = np.concatenate((below_bounds, above_bounds))
        rvs_less = np.delete(rvs, bad)
        if len(bad) > 4:
            rvs_add = self.distfit.rvs(size=len(bad))
        else:
            rvs_add = np.random.uniform(low=self.dmin, high=self.dmax, size=len(bad))
        
        rvs_ = np.concatenate((rvs_add, rvs_less))
        rvs_ = np.random.permutation(rvs_)
        
        return rvs_, len(rvs_add)


    def get_rvs(self, size=100):
        """Returns random variables using fitted continuous distribution"""
        rvs = self.distfit.rvs(size=size)

        rvs, add_num = self._cut_tails(rvs)
        while add_num > 4:
            # print "cut tails and fill up"
            rvs, add_num = self._cut_tails(rvs)

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


def cut_longtail(dist):
    terminate_tail = 1e-4
    pdf_dmax = dist.distfit.pdf(dist.dmax)
    try: pdf_dmax_ = pdf_dmax[0]
    except: pdf_dmax_ = pdf_dmax
    if pdf_dmax_ < terminate_tail:
        find_tail_end = lambda x: terminate_tail - dist.distfit.pdf(x)
        x_end = so.fsolve(find_tail_end, dist.dmax*0.5)
        return x_end
    else:
        return dist.dmax





def plot_param_distributions(distlist, xlabels, npoints=1000, figsize=(16, 10), 
                             original_bars=True, generated_bars=True, cut_tail=False):
    fig = plt.figure(figsize=figsize)
    subplot_base = get_subplotnum(len(distlist))
    subplots = [int(subplot_base + str(i+1)) for i in range(len(distlist))]
    for subplot, dist, xlabel in zip(subplots, distlist, xlabels):
        # print "dist.bounds:", len(dist.bounds)
        # print "dist min, max:", dist.dmin, dist.dmax
        # print "dist.dmin:", dist.dmin
        rvs = dist.get_rvs(size=npoints)
        # print "len(rvs):", len(rvs)
        ax = fig.add_subplot(subplot)
        # ax.grid(True)
        w = dist.widths[0]
        # print 'w:', w
        # bounds = dist.bounds - w*0.25
        if generated_bars:
            ax.hist(rvs, bins=dist.bounds-w*0.25, normed=1, rwidth=0.5, 
                    color='lightsteelblue', lw=0, zorder=1) # 'aquamarine'  'lightblue'
        # ppx = np.linspace(0, dist.dmax, npoints)
        ppx = np.linspace(dist.dmin, dist.dmax, npoints)
        ppy = dist.distfit.pdf(ppx)
        if original_bars:
            ax.bar(dist.bounds[:-1]+w*0.5, dist.probs, w*0.5, lw=0, 
                   color='cornflowerblue', alpha=1, zorder=2) # 'dodgerblue'
        distcolor = 'chocolate' # 'greenyellow' # 'limegreen' # 'cornflowerblue'
        ax.plot(ppx, ppy, color=distcolor, ls='--', lw=2, zorder=3)
        ax.fill_between(ppx, 0, ppy, facecolor=distcolor, zorder=0, alpha=0.1)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, None)
        # ax.set_xlim(0, dist.dmax)
        backstep = w*0.5 if dist.dmin > 0.2 else 0 # dirty fix for nice plotting
        dmax_ = cut_longtail(dist) if cut_tail else dist.dmax
        ax.set_xlim(dist.dmin-backstep, dmax_)
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
        distdict = rdb.loadObject('./asteroid_data/param_dist.p')

    #     if params is None:
    #         params = rdb.loadObject('./asteroid_data/orbparams_minmax.p')
    #     rand_params = ({name:np.random.uniform(low=values[0], high=values[1], 
    #                     size=num) for name, values in params.items()})
    # else:

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

def gen_rand_orbits(names, distlist, num=100):
    distdict = {name:dist for name, dist in zip(names, distlist)}
    rand_params = gen_rand_params(distdict=distdict, num=num)
    names_extend = rand_params.keys()
    randdata = np.array([rand_params[name] for name in names_extend]).T
    dataframe = pd.DataFrame(randdata, columns=names_extend)
    return dataframe


def rgen_orbits(distdict, num, rand_params=None, ri=0):
    if rand_params is None:
        rand_params = ({name: contdist.get_rvs(size=num)
                        for name, contdist in distdict.items()})
    else:
        for name, contdist in distdict.items():
            add_rvs = contdist.get_rvs(size=num)
            rand_params[name] = np.concatenate((rand_params[name], add_rvs))

    rand_params['e'] = (rand_params['a'] - rand_params['q'])/rand_params['a']

    e_rand = rand_params['e']
    n_neg = len(e_rand[e_rand < 0])
    print "n_neg:", n_neg
    # if n_neg == 0:
    #     print "good"
    #     names_extend = rand_params.keys()
    #     randdata = np.array([rand_params[name] for name in names_extend]).T
    #     dataframe = pd.DataFrame(randdata, columns=names_extend)
    #     print "len(dataframe):", len(dataframe)
        # return dataframe
    if ri > 50: 
        print "too high number of iterations has been reached:", ri
        return None
    elif n_neg > 0:
        rand_params_ = {name:  list() for name in rand_params}
        for i, e in enumerate(e_rand):
            if e >= 1.0:
                # warnings.warn(msg_ehigh)
                print msg_ehigh
                rand_params['e'][i] = 0.99
            elif e > 0:
                for name in rand_params:
                    rand_params_[name].append(rand_params[name][i])
        del rand_params
        rand_params = {name: np.asarray(rvs_list) for name, rvs_list in rand_params_.items()}
        del rand_params_

        print "len(rand_params['a']):", len(rand_params['a'])
        
        ri += 1
        rand_params = rgen_orbits(distdict, n_neg, rand_params, ri)
    # return dataframe
    # else:
    # if n_neg == 0:
    return rand_params
    # names_extend = rand_params.keys()
    # randdata = np.array([rand_params[name] for name in names_extend]).T
    # dataframe = pd.DataFrame(randdata, columns=names_extend)
    # return dataframe






def gen_orbits(distdict, num=100):
    # distdict = {name:dist for name, dist in zip(names, distlist)}
    # rand_params = gen_rand_params(distdict=distdict, num=num)
    rand_params = ({name: contdist.get_rvs(size=num)
                    for name, contdist in distdict.items()})

    for i, e in enumerate(rand_params['e']):
        if e >= 1.0:
            warnings.warn('too high eccentricity is found. value has been reset to 0.99')
            print 'too high eccentricity is found. value has been reset to 0.99'
            rand_params['e'][i] = 0.99

    # rand_params['a'] = rand_params['q']/(1.0 - rand_params['e'])
    rand_params['e'] = (rand_params['a'] - rand_params['q'])/rand_params['a']

    e_rand = rand_params['e']

    e_rand_neg = e_rand[e_rand < 0]

    re


    names_extend = rand_params.keys()
    randdata = np.array([rand_params[name] for name in names_extend]).T
    dataframe = pd.DataFrame(randdata, columns=names_extend)
    return dataframe




def gen_orbits_inout(dist_common, dist_inner, dist_outer, bound=1.0, num=100):

    rand_params = ({name: cdist.get_rvs(size=num)
                            for name, cdist in dist_common.items()})

    q_rand = rand_params['q']
    num_in = len(q_rand[q_rand <= 1.0])
    num_out = len(q_rand[q_rand > 1.0])

    print "num_in:", num_in
    print "num_out:", num_out

    w_in = dist_inner['w'].get_rvs(size=num_in)
    w_out = dist_outer['w'].get_rvs(size=num_out)

    w_in = np.random.permutation(w_in)
    w_out = np.random.permutation(w_out)

    # print "len w_in:", len(w_in) #w_in.shape
    # print "len w_out:", len(w_out) #w_out.shape

    # rand_params['a'] = np.zeros(num)
    rand_params['w'] = np.zeros(num)
    i_in = i_out = 0
    for i, q, e in zip(range(num), rand_params['q'], rand_params['e']):
        # just in case to avoid possible surprises
        if rand_params['e'][i] >= 1.0:
            warnings.warn('too high eccentricity is found. value has been reset to 0.99')
            rand_params['e'][i] = 0.99 

        if q <= 1.0:
            rand_params['w'][i] = w_in[i_in]
            i_in += 1
        else:
            # try
            rand_params['w'][i] = w_out[i_out]
            i_out += 1
    
    rand_params['a'] = rand_params['q']/(1.0 - rand_params['e'])

    # e_rand = rand_params['e']
    # print "e_rand[e_rand >= 1]:", e_rand[e_rand >= 0.9]

    # print len(rand_params['e']), type(rand_params['e'])
    # print len(rand_params['i']), type(rand_params['i'])
    # print len(rand_params['om']), type(rand_params['om'])
    # print len(rand_params['q']), type(rand_params['q'])
    # print len(rand_params['w']), type(rand_params['w'])
    # print len(rand_params['a']), type(rand_params['a'])

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
    # gen_rand_params(params=params)
    
    print "init orbit generation..."
    # names = ['a', 'e', 'i', 'w', 'om', 'q']
    # gkde = GaussianKDE('gkde', data_full['w'].as_matrix())
    # gkde2 = GaussianKDE('gkde2', data_full['om'].as_matrix())
    # gkde_a = GaussianKDE('gkde_a', data_full['a'].as_matrix())
    # kde_a = GaussianKDE(data_full['a'])
    names = ['a', 'i', 'w', 'om', 'q']
    bimod = BimodalDistribution()  # ss.logistic, ss.logistic
    statdists = [ss.johnsonsb, ss.exponweib, HarmonicDistribution(), HarmonicDistribution(), ss.pearson3] # ss.exponweib ss.loggamma
    # ss.genlogistic ss.exponweib ss.loggamma ss.burr
    # ss.fatiguelife ss.foldnorm ss.genpareto ss.gompertz!!!  ss.johnsonsb!!! ss.pearson3 ss.powerlognorm ss.recipinvgauss
    # ss.uniform, ss.beta
    data_full = pd.concat([haz[names], nohaz[names]])
    distlist = get_param_distributions(data_full, names, statdists, n=25, verbose=True)

    randdata = gen_rand_orbits(params, names, distlist, num=2e5)
    print "orbit generation finished."
    print "randdata sample:\n", randdata[:5]
    plot_param_distributions(distlist, names)
    
    # ### CALCULATE MOID ###
    # data = rdb.calc_moid(randdata, jobtime=True)
    # # haz, nohaz = rdb.get_hazMOID(data)

    # ### DUMP RANDOM ORBITS ###
    # haz_rand, nohaz_rand = rdb.get_hazMOID(randdata)
    # rdb.dumpObject(haz_rand, './asteroid_data/haz_rand_2e5m.p')
    # rdb.dumpObject(nohaz_rand, './asteroid_data/nohaz_rand_2e5m.p')
    # print "haz_rand:", len(haz_rand)
    # print "nohaz_rand:", len(nohaz_rand)

    # ### DUMP PARAMETERS DISTRIBUTIONS ###
    # distdict = {name: dist for name, dist in zip(names, distlist)}
    # rdb.dumpObject(distdict, './asteroid_data/param_dist.p')
    # # rdb.dumpObject(distlist, './asteroid_data/param_distlist.p')

    # rand_params = gen_rand_params(num=4)
    # # print "rand_params:", rand_params
    # # for key, value in rand_params.items():
    # #     print "%s\t%d" %(key, len(value))



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


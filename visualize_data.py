import string
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from matplotlib import cm
from copy import deepcopy
import learn_data as ld
# from string import split
# import matplotlib.colors as mpl_colors
# import matplotlib.cm as cmx
# from draw_ellipse_3d import OrbitDisplayGL
# from learn_data import loadObject

_axsize = [0.09, 0.05, 0.8, 0.9] # [left, bottom, width, height]
# _cbar_size = [0.9, 0.25, 0.03, 0.5]
_cbar_size = [0.9, 0.05, 0.04, 0.9]


colnames = {'a':  "Semi-major axis, AU",
            'q':  "Perihelion distance, AU",
            'i':  "Inclination, deg",
            'e':  "Eccentricity",
            'w':  "Argument of perihelion, deg",
            'om': "Longitude of the ascending node, deg"}

colnames_norm = {k: string.split(v, ',')[0] + ' (normalized)' for k, v in colnames.items()}

combs = list(itertools.combinations(['a', 'i', 'w', 'e', 'q', 'om'], 2))



def print_summary(extracted, trapped, haz_all, nohaz_all, dataname):
    haz_correct_num = sum(map(len, extracted))
    nohaz_trapped_num = sum(map(len, trapped))

    haz_correct_fraction = float(haz_correct_num)/len(haz_all)
    nohaz_trapped_fraction = float(nohaz_trapped_num)/len(nohaz_all)

    subgroup_num = nohaz_trapped_num + haz_correct_num
    haz_extracted_purity = float(haz_correct_num)/subgroup_num

    print "Number of correctly classified %s PHAs" % dataname, 
    print haz_correct_num
    print "Number of trapped %s NHAs:" % dataname, nohaz_trapped_num
    print
    print "Mass fraction of correctly classified %s PHAs:" % dataname, 
    print haz_correct_fraction
    print "Mass fraction of trapped %s NHAs:" % dataname, nohaz_trapped_fraction
    print
    print "Cummulative purity of the outlined PHA regions:", haz_extracted_purity  


def plot_alldistcombs(haz, nohaz, labels=True, plotsize=10):
    """
    Plots all combinations of pairs of orbital parameters for PHAs 
    and non-PHAs separately.
    """
    nplot = 0
    nrows = len(combs)
    fig = plt.figure(figsize=(2*plotsize, nrows*plotsize))
    for comb in combs:
        xname, yname = comb
        nplot = plot_distributions2d(list(comb), haz, nohaz, nplot, nrows, fig, 
                                     labels=labels)
        # print "nplot:", nplot
    plt.show()

def plot_distributions2d(cutcols, haz, nohaz, nplot=0, nrows=1, fig=None, 
                         figsize=(20,10), invertaxes=[0,0], colors=None,
                         s=5, alpha=1, labels=False, line=None):
    """
    Plots distributions of asteroids by two orbital parameters for PHAs
    and non-PHAs separately on different plots.
    """

    if labels:
        labels = [colnames[i] for i in cutcols]
    else: labels = None

    if colors is None:
        colors = ["orange", "blue"]

    haz_cut, nohaz_cut = ld.cut_params(haz, nohaz, cutcols)
    alldata = np.concatenate((haz_cut, nohaz_cut))
    xlim = [min(alldata[:,0]), max(alldata[:,0])]
    ylim = [min(alldata[:,1]), max(alldata[:,1])]

    show = False
    if fig is None:
        show = True
        fig = plt.figure(figsize=figsize)

    nplot += 1
    sp1 = (nrows, 2, nplot)
    nplot += 1
    sp2 = (nrows, 2, nplot)

    if line is not None:
        px, py = np.array(line).T

    ax1 = fig.add_subplot(*sp1)
    _add_distplot(ax1, nohaz_cut, colors[1], s, alpha, labels, xlim, ylim,
                  invertaxes=invertaxes)
    if line is not None: ax1.plot(px, py, lw=2, color='red')

    ax2 = fig.add_subplot(*sp2)
    _add_distplot(ax2, haz_cut, colors[0], s, alpha, labels, xlim, ylim,
                  invertaxes=invertaxes)
    if line is not None: ax2.plot(px, py, lw=2, color='red')

    if show:
        plt.show()
    else:
        return nplot

def plot_distribution2d(cutcol, haz=None, nohaz=None, labels=False,
                        invertaxes=[0,0], figsize=(8,8),
                        colors=None, grid_color=None, s=5, 
                        alpha=1, gridlines=True, xlines=None, ylines=None):
    
    """
    Plots distribution of asteroids by two orbital parameters for PHAs
    and non-PHAs on the same plot.
    """

    if labels:
        labels = [colnames[i] for i in cutcol]
    else: labels = None

    if colors is None:
        colors = ["orange", "blue"]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    haz_cut, nohaz_cut = ld.cut_params(haz, nohaz, cutcol)
    # print "map(type, [haz_cut, nohaz_cut]):", map(type, [haz_cut, nohaz_cut])
    single = lambda a: a[0] if a[1] is None else a[1]
    if type(None) not in map(type, [haz, nohaz]):
        alldata = np.concatenate((haz_cut, nohaz_cut))
    else:
        alldata = single([haz_cut, nohaz_cut])
        # print "alldata:", alldata

    xlim = [min(alldata[:,0]), max(alldata[:,0])]
    ylim = [min(alldata[:,1]), max(alldata[:,1])]

    if nohaz is not None:
        _add_scatterplot(ax, nohaz_cut, colors[1], s, alpha)
    if haz is not None:
        _add_scatterplot(ax, haz_cut, colors[0], s, alpha)

    _adjust_axes(ax, labels, xlim, ylim, invertaxes, grid_color, gridlines,
                 xlines, ylines)
    plt.show()

def plot_distribution3d(haz, nohaz):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(haz[..., 0], haz[..., 1], haz[..., 2], s=10, c="red", lw=0)
    ax.scatter(nohaz[..., 0], nohaz[..., 1], nohaz[..., 2], s=10, c="blue", lw=0)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def plot_densclusters(datasets, labels=None, scales=None, bgcolor='black',
                      invertaxes=[0, 0], figsize=(8,8), cmap='winter', 
                      grid_color=None, gridlines=False, s=3, alpha=1):
    
    if scales is None:
        scales = [(0,1), (0,1)]

    fig = plt.figure(figsize=figsize)
    # bgcolor = (0.05, 0.06, 0.14)
    # bgcolor = (0.0, 0.0, 0.0)
    # bgcolor = "navy"
    # fig.patch.set_facecolor(bgcolor)
    
    # ax = fig.add_subplot(111)
    ax = fig.add_axes(_axsize)
    ax.set_axis_bgcolor(bgcolor)
    

    colors = get_colorlist(len(datasets), cmap=(cmap+'_r'))
    for color, dataset in zip(colors, datasets):
        _add_scatterplot(ax, dataset, color, s, alpha, scales=scales)
    
    xlim, ylim = map(list, scales)
    _adjust_axes(ax, labels=labels, xlim=xlim, ylim=ylim, invertaxes=invertaxes,
                 grid_color=grid_color, gridlines=gridlines)

    cbax = fig.add_axes(_cbar_size)
    _add_colorbar(cbax, len(datasets), cmap, 'DB cluster ID')
    # rescale_axes(ax, scales)
    # ax.invert_yaxis()
    plt.show()

def _add_distplot(ax, dataset, color, s, alpha, labels, xlim, ylim, invertaxes,
                  gridcolor=None, gridlines=True, xlines=None, ylines=None):
    
    _add_scatterplot(ax, dataset, color, s, alpha)
    _adjust_axes(ax, labels, xlim, ylim, invertaxes, gridcolor, 
                 gridlines, xlines, ylines)

def _add_scatterplot(ax, dataset, color, s, alpha, scales=None):

    if scales is None:
        xmin, xmax = 0, 1
        ymin, ymax = 0, 1
    else:
        xmin, xmax = scales[0]
        ymin, ymax = scales[1]

    xdata = dataset[..., 0]*(xmax - xmin) + xmin
    ydata = dataset[..., 1]*(ymax - ymin) + ymin
    ax.scatter(xdata, ydata, s=s, c=color, lw=0, alpha=alpha)

def _add_colorbar(ax, ncolors, cm, label, values=None):

    cmap = mpl.cm.get_cmap(cm, ncolors)
    # colors = get_colorlist(ncolors, cmap=cm)
    # print "colors:", colors
    # print "cmap:", cmap
    # norm_color = mpl.colors.Normalize(vmin=2, vmax=5) # clip=True
    # bounds=[0,1,2,3,4]
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # print "cmap:", cmap
    ticks = range(ncolors + 1)
    bounds = np.array(ticks) + 0.5
    # print "bounds:", bounds
    # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, boundaries=bounds,
                                   ticks=ticks, spacing='uniform',
                                   label=label, orientation='vertical')
    # cbax.yaxis.set_ticklabels(['c%d' %i for i in ticks[::-1]])
    # cb.clim(-3.8, 0.3)
    if values is None:
        values = ['c%d' %i for i in ticks]
    ax.yaxis.set_ticklabels(values[::-1])

def _adjust_axes(ax, labels=None, xlim=None, ylim=None, invertaxes=[0,0], 
                 grid_color=None, gridlines=True, xlines=None, ylines=None):
    # ax.locator_params(axis='y',nbins=10)
    if xlines is not None:
        ax.locator_params(axis='x', nbins=xlines)
    if ylines is not None:
        ax.locator_params(axis='y', nbins=ylines)

    if len(xlim) >= 2:
        ax.set_xlim([min(xlim), max(xlim)])
    if len(ylim) >= 2:
        ax.set_ylim([min(ylim), max(ylim)])

    if labels is not None:
        ax.set_ylabel(labels[1])
        ax.set_xlabel(labels[0])

    if invertaxes[0]:
        ax.invert_xaxis()
    if invertaxes[1]:
        ax.invert_yaxis()

    if grid_color is not None:
        xlines = ax.get_xgridlines()
        ylines = ax.get_ygridlines()
        [xl.set_color(grid_color) for xl in xlines]
        [yl.set_color(grid_color) for yl in ylines]
        [xl.set_linewidth(3) for xl in xlines]
        [yl.set_linewidth(3) for yl in ylines]

    ax.grid(gridlines)
        
def _get_datagrid(x, y, num):
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, num),
                         np.linspace(ymin, ymax, num))
    return xx, yy

def _reverse_cmap(cmap_name):
    if cmap_name[-2:] == '_r':
        return cmap_name[:-2]
    else:
        return (cmap_name + '_r')

def get_colorlist(n, cmap='winter_r'):
    color_norm  = mpl.colors.Normalize(vmin=0, vmax=n-1)
    scalar_map = mpl.cm.ScalarMappable(norm=color_norm, cmap=cmap) 
    colors_list = [scalar_map.to_rgba(index) for index in range(n)]
    return colors_list


def plot_clf2d(clf, cutcol, num=1e2, haz_cut=None, nohaz_cut=None, labels=True, 
               clustprobs=None, rescale=True, scales=None, pointcolors=None,
               subgroups=None, gridcolor=None, plotbounds=None, invertaxes=[0, 0],
               figsize=(10,10), cmap='winter_r', s=5, alpha=1, gridlines=False, 
               extend_factors=None):
    
    """
    Plots decision boundaries of the classifier, trained for a 2-dimentional input data. 
    Optionally, plots distributions of asteroids by two orbital parameters for PHAs
    and non-PHAs.
    """

    if labels:
        labels = [colnames[i] for i in cutcol]
    else: labels = None

    if scales is None:
        scales = [(0,1), (0,1)]

    # if plotbounds is None:
    #     plotbounds = np.array([[0.0, 0.0], [1.0, 1.0]])

    if extend_factors is None:
        plotbounds = np.array([[0.0, 0.0], [1.0, 1.0]])
        efx = efy = 0
    else:
        efx, efy = extend_factors
        xmin, xmax = scales[0]
        ymin, ymax = scales[1]
        cx = efx * xmax / (xmax * (1.0 + 2 * efx))
        cy = efy * ymax / (ymax * (1.0 + 2 * efy))
        plotbounds = np.array([[cx, cy], [1-cx, 1-cy]])

    fig = plt.figure(figsize=figsize)
    if (clustprobs is not None) or (subgroups is not None):
        ax = fig.add_axes(_axsize)
        cbax = fig.add_axes(_cbar_size)
    else:
        ax = fig.add_subplot(111)

    # plot classifier
    xmin, xmax = plotbounds.T[0]
    ymin, ymax = plotbounds.T[1]
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, num),
                         np.linspace(ymin, ymax, num))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    levels = len(np.unique(z))
    zz = (z-levels).reshape(xx.shape)

    if rescale:
        xx, yy = _get_datagrid(scales[0], scales[1], num)

    ax.pcolor(xx, yy, zz, cmap=cmap)
    # ax.contourf(xx, yy, zz, cmap=cmap, norm=norm_color)

    # plot colorbar for class IDs
    if clustprobs is not None:
        probs = ["%.2f" % cp for cp in clustprobs]
        cb_title = 'Mass fraction of hazardous asteroids'
        _add_colorbar(cbax, len(np.unique(z)), _reverse_cmap(cmap), cb_title, 
                      values=probs)
    elif subgroups is not None:
        cb_title = 'Subgroups'
        # values = ['sg%d' %i for i in range(len(np.unique(z)))]
        _add_colorbar(cbax, len(np.unique(z)), cmap, cb_title, values=subgroups)
        # plt.colorbar(ax)

    # plot scatters
    if pointcolors is None:
        pointcolors = ["yellow", "darkblue"]

    sx, sy = scales
    efx_ = efx * (sx[1] - sx[0])
    efy_ = efy * (sy[1] - sy[0])
    pscales = [(sx[0]-efx_, sx[1]+efx_), (sy[0]-efy_, sy[1]+efy_)]

    if nohaz_cut is not None:
        _add_scatterplot(ax, nohaz_cut, pointcolors[1], s, alpha, scales=pscales)
    if haz_cut is not None:
        _add_scatterplot(ax, haz_cut, pointcolors[0], s, alpha, scales=pscales)

    xlim = [xx.min(), xx.max()]
    ylim = [yy.min(), yy.max()]
    _adjust_axes(ax, labels, xlim, ylim, invertaxes, gridcolor, gridlines)
    
    plt.show()

def plot_clf3d(clf, cutcol, plotbounds=None, num=1e2, haz=None, nohaz=None, labels=True, 
                    rescale=True, scales=None, invertaxes=[0, 0], figsize=(10,10),
                    cmap=cm.jet, grid_color=None, mode='2d', clf_masks=None, 
                    mask_level='above'):
    """
    Plots desicion surface of the classifier trained for a 3-dimensional input data.
    """

    if labels:
        labels = [colnames[i] for i in cutcol]
        labels_xy = labels[:2]
        cb_title = labels[2]
    else: 
        labels_xy = None
        cb_title = None

    fig = plt.figure(figsize=figsize)
    if mode == '3d':
        ax = Axes3D(fig)
        # ax = fig.add_subplot(111, projection='3d')
    elif mode == '2d':
        # ax = fig.add_subplot(111)
        ax = fig.add_axes(_axsize)
        cbax = fig.add_axes(_cbar_size)
    else:
        raise AttributeError("'mode' attribute must be one of '2d' and '3d'.")

    if plotbounds is not None:
        xb, yb, zb = plotbounds
    else:
        xb = yb = zb = [0.0, 1.0]

    xx, yy = _get_datagrid(xb, yb, num)
    x_ = xx.ravel()
    y_ = yy.ravel()

    z0 = np.zeros(len(x_))
    zm = deepcopy(z0)
    zlayers = np.linspace(zb[0], zb[1], num)
    # if scales is not None:
    #     zlayers = _rescale(zlayers, scales[2])
    # zsc = max(zlayers) - min(zlayers)

    # apply mask for the points that belong to the subgroup
    if clf_masks is not None:
        for clfm in clf_masks:
            clf_, v = clfm
            c = clf_.predict(np.c_[x_, y_])
            ccut = np.where(c == v)[0]
            zm[ccut] = -1

        # pni = np.where(z0 < 0)[0]
        ppi = np.where(zm == 0)[0]
        xp_ = x_[ppi]
        yp_ = y_[ppi]
        # zp_ = z0[ppi]
        z0_ = np.zeros(len(xp_))
    else:
        # xp_, yp_, zp = x_, y_, z0
        xp_, yp_ = x_, y_
        z0_ = z0

    # extract clf-outlined surface layer by layer
    for zi in zlayers:
        # print "zi:", zi
        zp_ = np.empty(len(yp_))
        # z_ = np.full((len(y_),), zi)
        zp_.fill(zi)

        c = clf.predict(np.c_[xp_, yp_, zp_])
        c1i = np.where(c == 1)[0]
        c0i = np.where(c == 0)[0]

        zp_[c0i] = zp_[c0i] - 2.0
        zp_ = np.maximum(zp_, z0_)
        z0_ = zp_

    if clf_masks is not None:
        z_ = deepcopy(z0)
        # pni = np.where(z0 < 0)[0]
        z_[ppi] = zp_

    else:
        z_ = zp_

    # rescale datapoints
    if scales is not None:
        xxs, yys = _get_datagrid(scales[0], scales[1], num)
        xs_, ys_, zs_ = [_rescale(ar, scales[i]) for i, ar in enumerate([x_, y_, z_])]
        levels = np.linspace(np.min(zs_), np.max(zs_), 100)
    else:
        xxs, yys = xx, yy
        xs_, ys_, zs_ = x_, y_, z_
        levels = np.linspace(0.0, 1.0, 100)
    
    # lower points not belonging to the subgroup
    if clf_masks is not None:
        pni = np.where(zm < 0)[0]
        # zs_[pni] = 5
        # below = min(zs_) * 2 - max(zs_)
        # below = max(zs_) + 2
        
        if mask_level == 'above':
            exclude = max(zs_) + 1
        elif mask_level == 'below':
            exclude = min(zs_) - 1
        else:
            raise AttributeError("mask_level argument must by one of 'above' or 'below'.")

        zs_[pni] = exclude

    zzs = zs_.reshape(xx.shape)
    _adjust_axes(ax, labels=labels_xy, xlim=[xxs.min(), xxs.max()], 
                 ylim=[yys.min(), yys.max()], invertaxes=invertaxes, 
                 grid_color=grid_color, gridlines=True)
    # ax.grid(True)

    if mode == '3d':
        # ax.plot_surface(xx, yy, zz)
        ax.plot_trisurf(xs_, ys_, zs_, cmap=cmap, linewidth=0)
    else:
        # levels = np.linspace(np.min(zzs), np.max(zzs), 100)
        # levels = np.linspace(0, np.max(zzs), 100)
        mpp = ax.contourf(xxs, yys, zzs, cmap=cmap, levels=levels) # cm.jet levels=levels
        # mpp = ax.pcolor(xxs, yys, zzs, cmap=cmap)
        # _add_colorbar(cbax, len(np.unique(z_)), cmap, cb_title)
        # cb = mpl.colorbar.ColorbarBase(cbax, cmap=cmap, orientation='vertical', label=cb_title)
        # cb = ax.add_cbar()
        plt.colorbar(mappable=mpp, cax=cbax, ax=ax, label=cb_title)

    plt.show()

def _rescale(data, scales):
    xmin, xmax = scales
    data_sc = data * float(xmax-xmin) + xmin
    return data_sc

def plot_scatter_clf3d(clf, plotbounds=None, num=1e2, haz=None, nohaz=None, labels=None, 
                    clustprobs=None, rescale=True, scales=[(0,1), (0,1)],
                    invertaxes=[0, 0], figsize=(10,10), cmap='winter_r', 
                    colors=["yellow","darkblue"], norm_color=None, subgroups=None,
                    grid_color=None):
    
    fig = plt.figure(figsize=figsize)
    # ax = fig.add_subplot(111)
    ax = fig.add_subplot(111, projection='3d')

    if plotbounds is not None:
        xb, yb, zb = plotbounds
    else:
        xb = yb = zb = [0.0, 1.0]

    xs = np.array([])
    ys = np.array([])
    zs = np.array([])

    zlayers = np.linspace(zb[0], zb[1], num)
    for zi in zlayers:
        xx, yy = _get_datagrid(xb, yb, num)
        x_ = xx.ravel()
        y_ = yy.ravel()
        z_ = np.empty(len(y_))
        # z_ = np.full((len(y_),), zi)
        z_.fill(zi)

        c = clf.predict(np.c_[x_, y_, z_])
        c1i = np.where(c==1)
        
        x1 = x_[c1i]
        y1 = y_[c1i]
        z1 = z_[c1i]

        xs = np.concatenate((xs, x1))
        ys = np.concatenate((ys, y1))
        zs = np.concatenate((zs, z1))

    ax.scatter(xs, ys, zs, c='r', marker='o', lw=0, alpha=1.0, s=10)
    # plt.clim(-3.8, 0.3)
    plt.show()

def plot_onegroup(dataset, clf, lev, levels, labels=None, fig=None, subplot=111, 
                  figsize=(10,10), cmap='CMRmap_r', num=100, show=True):
    
    if fig is None:
        fig = plt.figure(figsize=figsize) 
    ax = fig.add_subplot(subplot)
    xx, yy = _get_datagrid(dataset[:,0], dataset[:,1], num)
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z_hazind = np.where(z == 1.0)
    z[z_hazind] = lev
    zz = z.reshape(xx.shape)
    # ax1.contourf(xx, yy, zz, cmap=cmap, norm=mpl.colors.Normalize(vmin=-3.8, vmax=0.0))
    # ax1.pcolor(xx, yy, zz, cmap=cmap, vmin=-4, vmax=0.0)
    ax.pcolor(xx, yy, zz, cmap=cmap, vmin=levels[0], vmax=levels[1])
    ax.set_xlim([xx.min(), xx.max()])
    ax.set_ylim([yy.min(), yy.max()])
    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

    # plt.clim(-3.8,0.0)

    xlines = ax.get_xgridlines()
    ylines = ax.get_ygridlines()
    [xl.set_color('grey') for xl in xlines]
    [yl.set_color('grey') for yl in ylines]
    [xl.set_linewidth(3) for xl in xlines]
    [yl.set_linewidth(3) for yl in ylines]

    ax.grid(True)
    if show:
        plt.show()

def plot_kde(kde, levnum=4, numpoints=101, figsize=(10,10), scales=[(0,1), (0,1)]):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    grid = np.linspace(0,1,101)
    X, Y = np.meshgrid(grid, grid)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    Z = kde.score_samples(xy).reshape(X.shape)
    levels = np.linspace(0, Z.max(), levnum)
    X_, Y_ = _get_datagrid(scales[0], scales[1], 101)
    
    ax.contourf(X_, Y_, Z, levels=levels, cmap=plt.cm.Reds)
    ax.grid(True)
    plt.show()

    return levels


### Orbits ###

def get_ellipse(a, e, npoints = 50):
    f = a*e
    b = a*sqrt(1 - e**2)

    xpoints = []
    ypoints = []
    angles = np.radians(np.linspace(0, 360, npoints))
    # print "angles:", angles

    for t in angles:
        x = a*cos(t) + f
        y = b*sin(t)
        xpoints.append(x)
        ypoints.append(y)

    return xpoints, ypoints

def plot_ellipses(dataset, color):

    dataset_a = dataset[['a']]
    dataset_e = dataset[['e']]
    adata = dataset_a.as_matrix()
    edata = dataset_e.as_matrix()

    for a, e in zip(adata, edata):
        xp, yp = get_ellipse(a, e)
        plt.plot(xp, yp, color=color)

def plot_earthorbit():
    xpoints = []
    ypoints = []
    angles = np.radians(np.linspace(0, 360, 50))
    # print "angles:", angles
    for t in angles:
        x = cos(t)
        y = sin(t)
        xpoints.append(x)
        ypoints.append(y)
    plt.plot(xpoints, ypoints, color="green", lw=2)

def plot_orbits2D(orb_original, orb_inclined, orb_rotated):
    fig = plt.figure()
    ax1 = fig.add_subplot(224)
    ax1.plot(orb_original[..., 0], orb_original[..., 2], c="blue")
    ax1.plot(orb_inclined[..., 0], orb_inclined[..., 2], c="lime")
    ax1.plot(orb_rotated[..., 0], orb_rotated[..., 2], c="red")
    ax1.set_aspect('equal')
    ax1.grid(True)

    ax2 = fig.add_subplot(222)
    ax2.plot(orb_original[..., 0], orb_original[..., 1], c="blue")
    ax2.plot(orb_inclined[..., 0], orb_inclined[..., 1], c="lime")
    ax2.plot(orb_rotated[..., 0], orb_rotated[..., 1], c="red")
    ax2.set_aspect('equal')
    ax2.grid(True)

    ax3 = fig.add_subplot(223)
    ax3.plot(orb_original[..., 1], orb_original[..., 2], c="blue")
    ax3.plot(orb_inclined[..., 1], orb_inclined[..., 2], c="lime")
    ax3.plot(orb_rotated[..., 1], orb_rotated[..., 2], c="red")
    ax3.set_aspect('equal')
    ax3.grid(True)

    # ax.set_zlabel('Z Label')
    plt.show()




### Deprecated ###

def display_allparams(datasets, combs, names, plotsize=10, commonscales=True):
    """
    Plots all combinations of pairs of orbital parameters for PHAs 
    and non-PHAs separately.
    """
    nplot = 0
    nrows = len(combs)
    fig = plt.figure(figsize=(2*plotsize, nrows*plotsize))
    for comb in combs:
        xname, yname = comb
        nplot = display_param2d(list(comb), [names[xname], names[yname]], 
                                datasets, nplot, nrows, fig, commonscales=commonscales)
        # print "nplot:", nplot
    plt.show()

def display_param2d(cols, labels, datasets_, nplot=0, nrows=1, fig=None, 
                    figsize=(20,10), invertaxes=[0,0], commonscales=True):
    """
    Plots distribution of pair of orbital parameters for PHAs
    and non-PHAs separately.
    """

    # datasets = ld.prepare_data(cutcol=cols, datasets=datasets_)
    # datasets_x = [datasets[i][:, :-1] for i in range(len(datasets_))]
    # haz_cut, nohaz_cut = datasets_x[:2]

    # haz_cut, nohaz_cut = ld.cut_2params(cols, datasets_)
    haz_cut, nohaz_cut = ld.cut_params(datasets_[0], datasets_[1], cols)

    if commonscales:
        xvals = np.concatenate((haz_cut[..., 0], nohaz_cut[..., 0]))
        yvals = np.concatenate((haz_cut[..., 1], nohaz_cut[..., 1]))
        xlims = [min(xvals), max(xvals)]
        ylims = [min(yvals), max(yvals)]
        lims = [xlims, ylims]
    else:
        lims = None

    show = False
    if fig is None:
        show = True
        fig = plt.figure(figsize=figsize)

    nplot += 1
    sp1 = (nrows, 2, nplot)
    nplot += 1
    sp2 = (nrows, 2, nplot)
    ax1 = fig.add_subplot(*sp1)
    plot_distribution(ax=ax1, haz=None, nohaz=nohaz_cut, labels=labels, 
                      invertaxes=invertaxes, lims=lims)
    plt.grid(True)
    ax2 = fig.add_subplot(*sp2)
    plot_distribution(ax=ax2, haz=haz_cut, nohaz=None, labels=labels, 
                      invertaxes=invertaxes, lims=lims)
    plt.grid(True)
    if show:
        plt.show()
    else:
        return nplot

def linearcut_plot(p1, p2, haz_cut, nohaz_cut, figsize=(20,10), 
                   commonscales=True, invertaxes=[0,0], labels=None):
    """
    Plots split line over the distribution of pair of orbital parameters for PHAs
    and non-PHAs separately.
    """
    px, py = np.array([p1, p2]).T
    fig = plt.figure(figsize=figsize)

    if commonscales:
        xvals = np.concatenate((haz_cut[..., 0], nohaz_cut[..., 0]))
        yvals = np.concatenate((haz_cut[..., 1], nohaz_cut[..., 1]))
        xlims = [min(xvals), max(xvals)]
        ylims = [min(yvals), max(yvals)]
        lims = [xlims, ylims]
    else:
        lims = None
    
    ax1 = fig.add_subplot(121)
    # xlim, ylim = [], []
    # _add_scatterplots(ax1, [nohaz_cut], ['blue'], xlim, ylim, [(0, 1), (0, 1)])
    plot_distribution(ax=ax1, haz=None, nohaz=nohaz_cut, labels=labels, 
                      invertaxes=invertaxes, lims=lims)
    ax1.plot(px, py, lw=2, color='red')
    # ax1.set_xlim([min(xlim), max(xlim)])
    # ax1.set_ylim([min(ylim), max(ylim)])
    plt.grid(True)
    
    ax2 = fig.add_subplot(122)
    # xlim, ylim = [], []
    # _add_scatterplots(ax2, [haz_cut], ['orange'], xlim, ylim, [(0, 1), (0, 1)])
    plot_distribution(ax=ax2, haz=haz_cut, nohaz=None, labels=labels, 
                      invertaxes=invertaxes, lims=lims)
    ax2.plot(px, py, lw=2, color='red')
    # ax2.set_xlim([min(xlim), max(xlim)])
    # ax2.set_ylim([min(ylim), max(ylim)])
    plt.grid(True)
    plt.show()


def plot_classifier(data, clf, num=1e2, haz=None, nohaz=None, labels=None, 
                    clustprobs=None, rescale=True, scales=[(0,1), (0,1)],
                    invertaxes=[0, 0], figsize=(10,10), cmap='winter_r', 
                    colors=["yellow","darkblue"], norm_color=None, subgroups=None,
                    grid_color=None):
    
    fig = plt.figure(figsize=figsize)
    if (clustprobs is not None) or (subgroups is not None):
        ax = fig.add_axes(_axsize)
        cbax = fig.add_axes(_cbar_size)
    else:
        ax = fig.add_subplot(111)

    xx, yy = _get_datagrid(data[:,0], data[:,1], num)
    # print "xx.shape:", xx.shape
    # print "yy.shape:", yy.shape
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # print "z.shape:", z.shape
    levels = len(np.unique(z))
    # print "levels:", levels
    # print np.unique(z)
    # print "(z-levels):\n", (z-levels)
    zz = (z-levels).reshape(xx.shape)
    # print np.unique(zz)
    # print "zz.shape:", zz.shape

    if rescale:
        xx, yy = _get_datagrid(scales[0], scales[1], num)
    # print "xx.shape:", xx.shape
    # print "yy.shape:", yy.shape
    # levels = np.arange(len(np.unique(z)))
    # levels = len(np.unique(z))
    if norm_color is not None:
        vmin, vmax = norm_color
        norm_color = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        ax.pcolor(xx, yy, zz, cmap=cmap, vmin=vmin, vmax=vmax)
        # ax.set_xlim([xx.min(), xx.max()])
        # ax.set_ylim([yy.min(), yy.max()])
        # plt.axis([xx.min(), xx.max(), yy.min(), yy.max()])
    # ax.contourf(xx, yy, zz, cmap=cmap, norm=norm_color) # alpha=0.8  plt.cm.winter_r
    else:
        ax.pcolor(xx, yy, zz, cmap=cmap)
        # ax.contourf(xx, yy, zz, cmap=cmap, norm=norm_color)
        # plt.axis([xx.min(), xx.max(), yy.min(), yy.max()])
    ax.set_xlim([xx.min(), xx.max()])
    ax.set_ylim([yy.min(), yy.max()])
    lims = ([xx.min(), xx.max()], [yy.min(), yy.max()])

    ax.grid(True)
    # ax.imshow(zz,interpolation='gaussian')
    # data = scipy.ndimage.zoom(data, 3)
    # ax.contour(xx, yy, zz, 2, lw=2, colors='k')
    # ax.imshow(zz, interpolation='bilinear', origin='lower', cmap='winter_r')
    if clustprobs is not None:
        probs = ["%.2f" % cp for cp in clustprobs]
        cb_title = 'Mass fraction of hazardous asteroids'
        _add_colorbar(cbax, len(np.unique(z)), _reverse_cmap(cmap), cb_title, 
                      values=probs)
    elif subgroups is not None:
        cb_title = 'Subgroups'
        # values = ['sg%d' %i for i in range(len(np.unique(z)))]
        _add_colorbar(cbax, len(np.unique(z)), cmap, cb_title, values=subgroups)
        # plt.colorbar(ax)

    plot_distribution(ax=ax, haz=haz, nohaz=nohaz, labels=labels, colors=colors,
                      invertaxes=invertaxes, scales=scales, grid_color=grid_color,
                      lims=lims)

    # plt.clim(-3.8, 0.3)
    plt.show()

def plot_distribution(ax=None, haz=None, nohaz=None, show=True, labels=None,
                      invertaxes=[0,0], scales=[(0, 1), (0, 1)], figsize=(12,10),
                      colors=["orange", "blue"], lims=None, grid_color=None):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        s, alpha = 4, 0.5
    else:
        s, alpha = 5, 1

    # alpha = 0.5

    xlim, ylim = [], []
    if nohaz is not None:
        _add_scatterplots(ax, [nohaz], [colors[1]], xlim, ylim, scales, alpha=alpha)
    if haz is not None:
        _add_scatterplots(ax, [haz], [colors[0]], xlim, ylim, scales, alpha=alpha)

    if lims is None:
        xlim_, ylim_ = xlim, ylim
    else:
        xlim_, ylim_ = lims
    _adjust_axes(ax, labels=labels, xlim=xlim_, ylim=ylim_, invertaxes=invertaxes, grid_color=grid_color)
    # _adjust_axes(ax, labels=labels, invertaxes=invertaxes)

    if ax is None:
        plt.show()

def _add_scatterplots(ax, datasets, colors, xlim, ylim, scales, alpha=1):
    xmin, xmax = scales[0]
    ymin, ymax = scales[1]
    for color, dataset in zip(colors, datasets):
        xdata = dataset[..., 0]*(xmax - xmin) + xmin
        ydata = dataset[..., 1]*(ymax - ymin) + ymin
        ax.scatter(xdata, ydata, s=6, c=color, lw=0, alpha=alpha) # , alpha=0.5
        xlim.append(min(xdata))
        xlim.append(max(xdata))
        ylim.append(min(ydata))
        ylim.append(max(ydata))

### Leftovers ###


# def plot_minigroups():

#     ax1 = fig.add_subplot(*sp1)
#     ax1 = fig.add_subplot(131)
#     xx, yy = vd._get_datagrid(xtrain_mg11[:,0], xtrain_mg11[:,1], num)
#     z = clf11.predict(np.c_[xx.ravel(), yy.ravel()])
#     # print np.unique(z)
#     z_hazind = np.where(z == 1.0)
#     z[z_hazind] = -4.
#     # z = z
#     # levels = len(np.unique(z))
#     # zz = (z-0.5).reshape(xx.shape)
#     zz = z.reshape(xx.shape)

#     # ax1.contourf(xx, yy, zz, cmap=cmap, norm=mpl.colors.Normalize(vmin=-3.8, vmax=0.0))
#     ax1.pcolor(xx, yy, zz, cmap=cmap, vmin=-4, vmax=0.0)
#     ax1.set_xlim([xx.min(), xx.max()])
#     ax1.set_ylim([yy.min(), yy.max()])
#     ax1.set_xlabel(labels[0])
#     ax1.set_ylabel(labels[1])
#     # plt.clim(-3.8,0.0)
#     ax1.grid(True)

# def rescale_axes(ax, scales):
    # ax.set_autoscaley_on(False)
    # ax.set_ylim([0,1000])
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # scale_x = 1e-9
    # ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
    # xscale, yscale = scales
    # if xscale is not None:
    #     ticks_x = ticker.FuncFormatter(lambda x, pos: '{}'.format((x-xscale[0])*(xscale[1] - xscale[0])))
    #     ax.xaxis.set_major_formatter(ticks_x)

    # xscale, yscale = scales
    # if xscale is not None:
    #     # ax.set_xticks([x/8.0 for x in range(9)])
    #     xlabels = ax.get_xticks().tolist()
    #     xlabels_ = [(val-xscale[0])*(xscale[1] - xscale[0]) for val in xlabels]
    #     ax.set_xticklabels(["%.2f" %f for f in xlabels_])
    # if yscale is not None:
    #     # ax.set_yticks([y/8.0 for y in range(9)])
    #     ylabels = ax.get_yticks().tolist()
    #     ylabels_ = [(val-yscale[0])*(yscale[1] - yscale[0]) for val in ylabels]
    #     ax.set_yticklabels(["%.2f" %f for f in ylabels_])

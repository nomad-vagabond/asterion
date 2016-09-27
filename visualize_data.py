import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
# from string import split
# import matplotlib.colors as mpl_colors
# import matplotlib.cm as cmx
# from draw_ellipse_3d import OrbitDisplayGL
# from learn_data import loadObject

_axsize = [0.09, 0.05, 0.8, 0.9]
_cbar_size = [0.9, 0.25, 0.03, 0.5]



def plot_classifier(data, clf, num=1e2, haz=None, nohaz=None, labels=None, 
                    clustprobs=None, scales=[(0,1), (0,1)], invertaxes=[0,1],
                    figsize=(10,10), cmap='winter_r', colors=["yellow","darkblue"]):
    
    fig = plt.figure(figsize=figsize)
    if clustprobs is not None:
        ax = fig.add_axes(_axsize)
        cbax = fig.add_axes(_cbar_size)
    else:
        ax = fig.add_subplot(111)

    xx, yy = _get_datagrid(data[:,0], data[:,1], num)
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    levels = len(np.unique(z))
    zz = (z-levels).reshape(xx.shape)

    xx, yy = _get_datagrid(scales[0], scales[1], num)
    # levels = np.arange(len(np.unique(z)))
    # levels = len(np.unique(z))
    ax.contourf(xx, yy, zz, cmap=cmap) # alpha=0.8  plt.cm.winter_r
    # ax.imshow(zz,interpolation='gaussian')
    # data = scipy.ndimage.zoom(data, 3)
    # ax.contour(xx, yy, zz, 2, lw=2, colors='k')
    # ax.imshow(zz, interpolation='bilinear', origin='lower', cmap='winter_r')
    if clustprobs is not None:
        probs = ["%.2f" % cp for cp in clustprobs]
        cb_title = 'Mass fraction of hazardous asteroids'
        _add_colorbar(cbax, len(np.unique(z)), _reverse_cmap(cmap), cb_title, 
                      values=probs)
    plot_distribution(ax=ax, haz=haz, nohaz=nohaz, labels=labels, colors=colors,
                      invertaxes=invertaxes, scales=scales)
    plt.show()

def plot_densclusters(datasets, labels=None, scales=[(0, 1), (0, 1)], 
                      invertaxes=[0,1], figsize=(12,10), cmap='winter'):
    # fig, ax = plt.subplots(figsize=figsize)
    fig = plt.figure(figsize=figsize)
    bgcolor = (0.05, 0.06, 0.14)
    # bgcolor = "navy"
    # fig.patch.set_facecolor(bgcolor)
    
    # ax = fig.add_subplot(111)
    ax = fig.add_axes(_axsize)
    ax.set_axis_bgcolor(bgcolor)
    cbax = fig.add_axes(_cbar_size)
    colors = get_colorlist(len(datasets))
    xlim, ylim = [], []

    _add_scatterplots(ax, datasets, colors, xlim, ylim, scales)
    _adjust_axes(ax, labels=labels, xlim=xlim, ylim=ylim, invertaxes=invertaxes)

    _add_colorbar(cbax, len(datasets), cmap, 'DB cluster ID')
    # rescale_axes(ax, scales)
    # ax.invert_yaxis()
    plt.show()

def plot_distribution(ax=None, haz=None, nohaz=None, show=True, labels=None,
                      invertaxes=[0,1], scales=[(0, 1), (0, 1)], figsize=(12,10),
                      colors=["orange", "blue"]):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        s, alpha = 4, 0.5
    else:
        s, alpha = 5, 1

    xlim, ylim = [], []
    if nohaz is not None:
        _add_scatterplots(ax, [nohaz], [colors[1]], xlim, ylim, scales, alpha=alpha)
    if haz is not None:
        _add_scatterplots(ax, [haz], [colors[0]], xlim, ylim, scales, alpha=alpha)
    _adjust_axes(ax, labels=labels, xlim=xlim, ylim=ylim, invertaxes=invertaxes)

    if ax is None:
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


def _add_colorbar(ax, ncolors, cm, label, values=None):
    cmap = mpl.cm.get_cmap(cm, ncolors)
    # print "cmap:", cmap
    ticks = range(ncolors + 1)
    bounds = np.array(ticks) + 0.5
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, boundaries=bounds,
                                   ticks=ticks, spacing='uniform',
                                   label=label, orientation='vertical')
    # cbax.yaxis.set_ticklabels(['c%d' %i for i in ticks[::-1]])
    if values is None:
        values = ['c%d' %i for i in ticks]
    ax.yaxis.set_ticklabels(values[::-1])

def _add_scatterplots(ax, datasets, colors, xlim, ylim, scales, alpha=1):
    xmin, xmax = scales[0]
    ymin, ymax = scales[1]
    for color, dataset in zip(colors, datasets):
        xdata = dataset[..., 0]*(xmax - xmin) + xmin
        ydata = dataset[..., 1]*(ymax - ymin) + ymin
        ax.scatter(xdata, ydata, s=5, c=color, lw=0, alpha=alpha) # , alpha=0.5
        xlim.append(min(xdata))
        xlim.append(max(xdata))
        ylim.append(min(ydata))
        ylim.append(max(ydata))

def _adjust_axes(ax, labels=None, xlim=None, ylim=None, invertaxes=[0,0]):
    # ax.locator_params(axis='y',nbins=10)
    # ax.locator_params(axis='x',nbins=10)

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


def _get_datagrid(x, y, num):
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    hx = float(xmax - xmin)/num
    hy = float(ymax - ymin)/num
    xx, yy = np.meshgrid(np.arange(xmin, xmax, hx),
                         np.arange(ymin, ymax, hy))
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

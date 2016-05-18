import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.colors as mpl_colors
# import matplotlib.cm as cmx
# from draw_ellipse_3d import OrbitDisplayGL
# from learn_data import loadObject


def get_colorlist(n, cmap='winter_r'):
    color_norm  = mpl.colors.Normalize(vmin=0, vmax=n-1)
    scalar_map = mpl.cm.ScalarMappable(norm=color_norm, cmap=cmap) 
    colors_list = [scalar_map.to_rgba(index) for index in range(n)]
    return colors_list




def plot_classifier(data, clf, num=1e2, haz=None, nohaz=None, labels=None,
                    clustprobs=None, scales=[(0,360), (0, 1)], figsize=(10,10)):
    fig = plt.figure(figsize=figsize)
    if clustprobs is not None:
        ax = fig.add_axes([0.05, 0.05, 0.8, 0.9])
        cbax = fig.add_axes([0.9, 0.2, 0.04, 0.6])
    else:
        ax = fig.add_subplot(111)

    xmin, xmax = np.min(data[:,0]), np.max(data[:,0])
    ymin, ymax = np.min(data[:,1]), np.max(data[:,1])
    hx = float(xmax - xmin)/num
    hy = float(ymax - ymin)/num
    xx, yy = np.meshgrid(np.arange(xmin, xmax, hx),
                         np.arange(ymin, ymax, hy))

    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    zz = z.reshape(xx.shape)
    ax.contourf(xx, yy, zz, cmap='winter_r') # alpha=0.8  plt.cm.winter_r
    if clustprobs is not None:
        add_colorbar(cbax, len(np.unique(z)), 'winter', 'Mass fraction of hazardous asteroids', values=clustprobs)
    rescale_axes(ax, scales)

    plot_distribution(ax=ax, haz=haz, nohaz=nohaz, labels=labels)
    plt.show()

    # randdata = np.array([np.random.uniform(low=xmin, high=xmax, size=num),
    #                      np.random.uniform(low=ymin, high=ymax, size=num),]).T
    # predict = clf.predict(randdata)

def add_colorbar(ax, ncolors, cm, label, values=None):
    # cn = len(ncolors)
    cmap = mpl.cm.get_cmap(cm, ncolors)
    ticks = range(ncolors + 1)
    bounds = np.array(ticks) + 0.5
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, boundaries=bounds,
                                   ticks=ticks, spacing='uniform',
                                   label=label, orientation='vertical')
    # cbax.yaxis.set_ticklabels(['c%d' %i for i in ticks[::-1]])
    if values is None:
        values = ['c%d' %i for i in ticks]
    ax.yaxis.set_ticklabels(values[::-1])



def plot_distribution3d(haz, nohaz):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(haz[..., 0], haz[..., 1], haz[..., 2], s=10, c="red", lw=0)
    ax.scatter(nohaz[..., 0], nohaz[..., 1], nohaz[..., 2], s=10, c="blue", lw=0)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def plot_densclusters(datasets, labels=None, scales=[(0,360), (0, 1)], figsize=(10,10)):
    # fig, ax = plt.subplots(figsize=figsize)
    fig = plt.figure(figsize=figsize)
    # ax = fig.add_subplot(111)
    ax = fig.add_axes([0.05, 0.05, 0.8, 0.9])

    cbax = fig.add_axes([0.9, 0.2, 0.04, 0.6])
    colors = get_colorlist(len(datasets))
    xlim = []
    ylim = []
    for color, dataset in zip(colors, datasets):
        ax.scatter(dataset[..., 0], dataset[..., 1], s=5, c=color, lw=0) # , alpha=0.5
        xlim.append(min(dataset[..., 0]))
        xlim.append(max(dataset[..., 0]))
        ylim.append(min(dataset[..., 1]))
        ylim.append(max(dataset[..., 1]))
    
    ax.set_xlim([min(xlim), max(xlim)])
    ax.set_ylim([min(ylim), max(ylim)])
    if labels is not None:
        ax.set_ylabel(labels[1])
        ax.set_xlabel(labels[0])

    add_colorbar(cbax, len(datasets), 'winter', 'DB cluster ID')
    rescale_axes(ax, scales)
    plt.show()



def rescale_axes(ax, scales):

    xscale, yscale = scales
    if xscale is not None:
        ax.set_xticks([x/8.0 for x in range(9)])
        xlabels = ax.get_xticks().tolist()
        xlabels_ = [(val-xscale[0])*(xscale[1] - xscale[0]) for val in xlabels]
        ax.set_xticklabels(xlabels_)
    if yscale is not None:
        ax.set_yticks([y/8.0 for y in range(9)])
        ylabels = ax.get_yticks().tolist()
        ylabels_ = [(val-yscale[0])*(yscale[1] - yscale[0]) for val in ylabels]
        ax.set_yticklabels(ylabels_)

    




def plot_distribution(ax=None, haz=None, nohaz=None, show=True, labels=None, figsize=(10,10)):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    # xmin, xmax = None, None
    # ymin, ymax = None, None
    xlim = []
    ylim = []

    if nohaz is not None:
        ax.scatter(nohaz[..., 0], nohaz[..., 1], s=3, c="blue", lw=0, alpha=0.5) # , alpha=0.5
        xlim.append(min(nohaz[..., 0]))
        xlim.append(max(nohaz[..., 0]))
        ylim.append(min(nohaz[..., 1]))
        ylim.append(max(nohaz[..., 1]))
        # xmin_nohaz, xmax_nohaz = min(nohaz[..., 0]), max(nohaz[..., 0])
        # ymin_nohaz, ymax_nohaz = min(nohaz[..., 1]), max(nohaz[..., 1])

    if haz is not None:
        ax.scatter(haz[..., 0], haz[..., 1], s=3, c="orange", lw=0, alpha=0.5) # , alpha=0.5
        xlim.append(min(haz[..., 0]))
        xlim.append(max(haz[..., 0]))
        ylim.append(min(haz[..., 1]))
        ylim.append(max(haz[..., 1]))
        # xmin_haz, xmax_haz = min(haz[..., 0]), max(haz[..., 0])
        # ymin_haz, ymax_haz = min(haz[..., 1]), max(haz[..., 1])
    
    if labels is not None:
        ax.set_ylabel(labels[1])
        ax.set_xlabel(labels[0])
    
    if len(xlim) >= 2:
        ax.set_xlim([min(xlim), max(xlim)])
    if len(ylim) >= 2:
        ax.set_ylim([min(ylim), max(ylim)])
    # plt.ylim([min(ymin_haz, ymin_nohaz), max(ymax_haz, ymax_nohaz)])
    # plt.xlim([min(xmin_haz, xmin_nohaz), max(xmax_haz, xmax_nohaz)])
    if ax is None:
        plt.show()

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


# if __name__ == '__main__':


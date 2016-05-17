import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from draw_ellipse_3d import OrbitDisplayGL
# from learn_data import loadObject


def plot_classifier(data, clf, num=1e2, haz=None, nohaz=None, labels=None, figsize=(10,10)):
    fig = plt.figure(figsize=figsize)
    xmin, xmax = np.min(data[:,0]), np.max(data[:,0])
    ymin, ymax = np.min(data[:,1]), np.max(data[:,1])
    hx = float(xmax - xmin)/num
    hy = float(ymax - ymin)/num
    xx, yy = np.meshgrid(np.arange(xmin, xmax, hx),
                         np.arange(ymin, ymax, hy))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plot_distribution(haz=haz, nohaz=nohaz, labels=labels, show=False)
    plt.show()

    # randdata = np.array([np.random.uniform(low=xmin, high=xmax, size=num),
    #                      np.random.uniform(low=ymin, high=ymax, size=num),]).T
    # predict = clf.predict(randdata)

def plot_distribution3d(haz, nohaz):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(haz[..., 0], haz[..., 1], haz[..., 2], s=10, c="red", lw=0)
    ax.scatter(nohaz[..., 0], nohaz[..., 1], nohaz[..., 2], s=10, c="blue", lw=0)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def plot_distribution(haz=None, nohaz=None, show=True, labels=None, figsize=(10,10)):
    if show:
        fig = plt.figure(figsize=figsize)
    # xmin, xmax = None, None
    # ymin, ymax = None, None
    xlim = []
    ylim = []

    if nohaz is not None:
        plt.scatter(nohaz[..., 0], nohaz[..., 1], s=3, c="blue", lw=0) # , alpha=0.5
        xlim.append(min(nohaz[..., 0]))
        xlim.append(max(nohaz[..., 0]))
        ylim.append(min(nohaz[..., 1]))
        ylim.append(max(nohaz[..., 1]))
        # xmin_nohaz, xmax_nohaz = min(nohaz[..., 0]), max(nohaz[..., 0])
        # ymin_nohaz, ymax_nohaz = min(nohaz[..., 1]), max(nohaz[..., 1])

    if haz is not None:
        plt.scatter(haz[..., 0], haz[..., 1], s=3, c="orange", lw=0) # , alpha=0.5
        xlim.append(min(haz[..., 0]))
        xlim.append(max(haz[..., 0]))
        ylim.append(min(haz[..., 1]))
        ylim.append(max(haz[..., 1]))
        # xmin_haz, xmax_haz = min(haz[..., 0]), max(haz[..., 0])
        # ymin_haz, ymax_haz = min(haz[..., 1]), max(haz[..., 1])
    
    if labels is not None:
        plt.ylabel(labels[1])
        plt.xlabel(labels[0])
    
    if len(xlim) >= 2:
        plt.xlim([min(xlim), max(xlim)])
    if len(ylim) >= 2:
        plt.ylim([min(ylim), max(ylim)])
    # plt.ylim([min(ymin_haz, ymin_nohaz), max(ymax_haz, ymax_nohaz)])
    # plt.xlim([min(xmin_haz, xmin_nohaz), max(xmax_haz, xmax_nohaz)])
    if show:
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


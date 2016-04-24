import numpy as np
from numpy.linalg import norm
import scipy.optimize as so
from math import cos, sin, sqrt, pi
# import matplotlib.pyplot as plt
from visualize_data import plot_orbits2D
from functools import partial
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d import proj3d

### Find Eart orbit ###

def get_earthorb_point(alpha): 

    a = 1.000001018
    e = 0.0167086
    i = np.radians(1.578690)
    w = np.radians(288.1)
    omega = np.radians(174.9)
    # get point on ellipse
    r = get_r(a, e, alpha)
    x = r*cos(alpha)
    y = r*sin(alpha)
    point = np.array([x, y, 0.0])
    # get inclined point
    ax = get_r(a, e, w)
    axis = np.array([ax*cos(w), ax*sin(w), 0])
    axis0 = axis/norm(axis)
    incpoint = rotate(point, axis0, i)
    # get rotated around Z point 
    angle = omega - w
    zaxis = np.array([0.0, 0.0, 1.0])
    rotpoint = rotate(incpoint, zaxis, angle)
    return rotpoint



### Fing perhelion of orbit projection to ecliptict ###

def get_r(a, e, t):
    r = a*(1 - e**2)/(1 + e*cos(t))
    return r

def get_orb_point(a, e, t): 
    r = get_r(a, e, t)
    x = r*cos(t)
    y = r*sin(t)
    return [x, y, 0]

def get_rotmatrix(ax, angle):
    cosa = cos(angle)
    sina = sin(angle)
    x, y, z = ax
    rot = np.array([[cosa + (x**2)*(1 - cosa), x*y*(1 - cosa) - z*sina, x*z*(1 - cosa) + y*sina],
                    [y*x*(1 - cosa) + z*sina, cosa + (y**2)*(1 - cosa), y*z*(1 - cosa) - x*sina],
                    [z*x*(1 - cosa) - y*sina, z*y*(1 - cosa) + x*sina, cosa + (z**2)*(1 - cosa)]])
    return rot

def rotate(point, ax, angle):
    rot = get_rotmatrix(ax, angle)
    return np.dot(point, rot)

def rotated_orb_point(a, e, w, i, omega, t):
    inc_point = inclined_orb_point(a, e, w, i, t)
    angle = omega - w
    zaxis = np.array([0.0, 0.0, 1.0])
    rotpoint = rotate(inc_point, zaxis, angle)
    return rotpoint

def inclined_orb_point(a, e, w, i, t):
    point = get_orb_point(a, e, t)
    r = get_r(a, e, w)
    axis = np.array([r*cos(w), r*sin(w), 0])
    axis0 = axis/norm(axis)
    rotpoint = rotate(point, axis0, i)
    return rotpoint

def get_rhor(t, a, e, w, i, omega):   
    rotpoint = rotated_orb_point(a, e, w, i, omega, t) 
    # rotpoint = inclined_orb_point(a, e, w, i, t)
    hpoint = rotpoint[:2]
    r_hor = norm(hpoint)
    return r_hor

def get_rver(t, a, e, w, i, omega):   
    rotpoint = rotated_orb_point(a, e, w, i, omega, t) 
    # rotpoint = inclined_orb_point(a, e, w, i, t)
    # vpoint = rotpoint[1:]
    # r_ver = norm(vpoint)
    r_ver = np.dot(rotpoint, np.array([0.0, 0.0, 1.0]))
    return r_ver

# def get_rmin(a, e, w, i, omega, direction):
#     if direction == 'horizontal':
#         func = partial(get_rhor, a=a, e=e, w=w, i=i, omega=omega)
#     elif direction == 'vertical':
#         func = partial(get_rver, a=a, e=e, w=w, i=i, omega=omega)
#     func_min = so.fmin(func, pi/2)
#     return func_min


def get_rmin(a, e, w, i, omega, direction):
    if direction == 'horizontal':
        func_min = so.fmin(partial(get_rhor, a=a, e=e, w=w, i=i, omega=omega), pi/2, disp=False)
    elif direction == 'vertical':
        func_min = so.fmin(partial(get_rver, a=a, e=e, w=w, i=i, omega=omega), pi/2, disp=False)
    # func_min = so.fmin(func, pi/2)
    return func_min

def get_rxry(a, e, w, i, omega):
    tmin = get_rmin(a, e, w, i, omega, direction="horizontal")
    rx = get_rhor(tmin, a, e, w, i, omega)
    # tmin = get_rmin(a, e, w, i, omega, direction="vertical")
    ry = get_rver(tmin, a, e, w, i, omega)
    return rx, ry

# def get_rw(a, e, w, i, omega):
#     tmin = get_rmin(a, e, w, i, omega, direction="horizontal")
#     rx = get_rhor(tmin, a, e, w, i, omega)
#     # tmin = get_rmin(a, e, w, i, omega, direction="vertical")
#     ry = get_rver(tmin, a, e, w, i, omega)
#     return rx, ry


# def get_earthorb_point(alpha):
#     x = cos(alpha)
#     y = sin(alpha)
#     return [x, y, 0]

def find_dist(ta, a, e, w, i, omega):
    epoint = get_earthorb_point(ta[1])
    # apoint = inclined_orb_point(a, e, w, i, ta[0])
    apoint = rotated_orb_point(a, e, w, i, omega, ta[0])
    dist = norm(apoint - epoint)
    return dist

def get_moid(a, e, w, i, omega):
    tmin = get_rmin(a, e, w, i, omega, direction="horizontal")
    tamin = so.fmin(partial(find_dist, a=a, e=e, w=w, i=i, omega=omega), [tmin, w], disp=False)
    moid = find_dist(tamin, a, e, w, i, omega)
    return moid


def find_center(a, e, w, i, omega):
    c = np.array([-1,0,0])*(a*e)
    r = get_r(a, e, w)
    axis = np.array([r*cos(w), r*sin(w), 0])
    axis0 = axis/norm(axis)
    c_inc = rotate(c, axis0, i)
    angle = omega - w
    zaxis = np.array([0.0, 0.0, 1.0])
    c_rot = rotate(c_inc, zaxis, angle)
    return c_rot


def project_center(c):
    cxy = norm(c[:2])

    # cz = norm(c[1:])
    cz = np.dot(c, np.array([0,0,1]))
    return cxy, cz

def get_rxrycxcy(a, e, w, i, omega):
    tmin = get_rmin(a, e, w, i, omega, direction="horizontal")
    rx = get_rhor(tmin, a, e, w, i, omega)
    # tmin = get_rmin(a, e, w, i, omega, direction="vertical")
    ry = get_rver(tmin, a, e, w, i, omega)
    c = find_center(a, e, w, i, omega)
    cx, cy = project_center(c)
    return rx, ry, cx, cy


### Find orbit points projections to ecliptics ###

def get_points(a, e, w, i, numpoints=100):
    # theta = np.linspace(0, 2*pi, numpoints)
    logspace_pi = np.logspace(0.01, 1, int(numpoints*0.5))*0.1
    lsp1 = (1 - logspace_pi)
    sp1 = np.sort(pi*(lsp1 - lsp1[0] + 1))
    lsp2 = (logspace_pi-logspace_pi[0])
    sp2 = lsp2*pi/lsp2[-1] + pi
    theta = np.concatenate((sp1,sp2[1:]))
    # dist = [get_r(t) for t in theta]
    points = np.array([get_orb_point(a, e, t) for t in theta])
    return points

def get_incpoints(a, e, w, i, points):
    r = get_r(a, e, w)
    axis = np.array([r*cos(w), r*sin(w), 0])
    axis0 = axis/norm(axis)
    rot = get_rotmatrix(axis0, i)
    inc_points = []
    for point in points:
        inc_point = np.dot(point, rot)
        inc_points.append(inc_point)
    return np.asarray(inc_points)

def get_rotpoints(a, e, w, i, omega, inc_points):
    angle = omega - w
    zaxis = np.array([0.0, 0.0, 1.0])
    rot = get_rotmatrix(zaxis, angle)
    rot_points = []
    for point in inc_points:
        rot_point = np.dot(point, rot)
        rot_points.append(rot_point)
    return np.asarray(rot_points)







if __name__ == '__main__':

    a = 1.07806413e+00
    e = 8.26926230e-01
    i = np.radians(2.28312083e+01)
    w = np.radians(3.13740962e+01)
    om = np.radians(8.80199731e+01)

    # a = 1.5
    # e = 0.9
    # i = -np.radians(30)
    # om = np.radians(0)
    # w = np.radians(150)
    # om = w


    # tmin = get_rmin(a, e, w, i, om, direction="horizontal")

    # rx = get_rhor(tmin, a, e, w, i, om)
    # ry = get_rver(tmin, a, e, w, i, om)
    rx, ry = get_rxry(a, e, w, i, om)
    print "rx, ry:", rx, ry

    moid = get_moid(a, e, w, i, om)
    print "moid:", moid


    # vmin = get_rmin(a, e, w, i, om, direction="vertical")
    # print "hmin, vmin:", hmin, vmin
    # print "get_rhor(0.086603, a, e, w, i, om):", get_rhor(0.086603, a, e, w, i, om)
    # print "get_rhor(0.0, a, e, w, i, om):", get_rhor(0.0, a, e, w, i, om)
    # print get_r1(0.186585)

    orb_points = get_points(a, e, w, i)
    inc_points = get_incpoints(a, e, w, i, orb_points)
    rot_points = get_rotpoints(a, e, w, i, om, inc_points)

    theta = np.linspace(0, 2*pi, 100)
    earthpoints = np.array([get_earthorb_point(t) for t in theta])

    # plot_orbits2D(orb_points, inc_points, rot_points)
    plot_orbits2D(orb_points, earthpoints, rot_points)

    # plt.plot(points, color="red")
    # plt.plot(points_rot, color="blue")
    # plt.grid(True)
    # plt.show()
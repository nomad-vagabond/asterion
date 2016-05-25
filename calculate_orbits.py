import random
import numpy as np
from numpy.linalg import norm
import scipy.optimize as so
from math import cos, sin, sqrt, pi, ceil
# import matplotlib.pyplot as plt
from visualize_data import plot_orbits2D
from functools import partial
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d import proj3d


Z_AXIS = np.array([0.0, 0.0, 1.0])

EARTH_A = 1.00000261
EARTH_E = 0.01671123
EARTH_I = 0.0
EARTH_W = np.radians(114.20783)
EARTH_OM = np.radians(348.73936)


def get_orbpoint_hc_direct(t, a, e, w, i, om):
    # returns point in heliocentric coord frame using in-plane angular parameter t and 3 orbit plane angles
    # distance to point from orbit focus
    r = _get_r(a, e, t)
    #heliocentric coords
    x = r * (cos(om) * cos(t + w) - sin(om) * sin(t + w) * cos(i))
    y = r * (sin(om) * cos(t + w) + cos(om) * sin(t + w) * cos(i))
    z = r * (sin(t + w) * sin(i))
    
    point = np.array([x, y, z])
    return point

def get_earthorbpoint_hc(t):
    return get_orbpoint_hc(EARTH_A, EARTH_E, EARTH_W, EARTH_I, EARTH_OM, t)

def get_orbpoint_hc(a, e, w, i, om, t):
    point = _get_orbpoint(a, e, t)             # point in orbital plane:
    axis_w = np.array([cos(-w), sin(-w), 0])
    point_inc = _rotate(point, axis_w, i)      # get inclined point:
    wb = om + w
    point_hc = _rotate(point_inc, Z_AXIS, wb)  # point in heliocentric coords:
    return point_hc

def _get_r(a, e, t):
    r = a*(1 - e**2)/(1 + e*cos(t))
    return r

def _get_orbpoint(a, e, t): 
    r = _get_r(a, e, t)
    x = r * cos(t)
    y = r * sin(t)
    return [x, y, 0]

def _get_rotmatrix(ax, angle):
    cosa = cos(angle)
    sina = sin(angle)
    x, y, z = ax
    rot = np.array([[cosa + (x**2)*(1 - cosa), x*y*(1 - cosa) - z*sina, x*z*(1 - cosa) + y*sina],
                    [y*x*(1 - cosa) + z*sina, cosa + (y**2)*(1 - cosa), y*z*(1 - cosa) - x*sina],
                    [z*x*(1 - cosa) - y*sina, z*y*(1 - cosa) + x*sina, cosa + (z**2)*(1 - cosa)]])
    return rot

def _rotate(point, ax, angle):
    rot = _get_rotmatrix(ax, angle)
    return np.dot(point, rot)

def _find_dist(t, a, e, w, i, om):
    # earth_point = get_earthorbpoint_hc(ta[1])
    # apoint = inclined_orb_point(a, e, w, i, ta[0])
    # apoint = rotated_orb_point(a, e, w, i, omega, ta[0])
    # asteroid_point = get_orbpoint_hc(a, e, w, i, om, t[0])
    # earth_point = get_earthorbpoint_hc(t[1])

    asteroid_point = get_orbpoint_hc_direct(t[0], a, e, w, i, om)
    earth_point = get_orbpoint_hc_direct(t[1], EARTH_A, EARTH_E, EARTH_W, EARTH_I, EARTH_OM)
    # earth_point = get_orbpoint_hc(EARTH_A, EARTH_E, EARTH_W, EARTH_I, EARTH_OM, t[1])
    dist = norm(asteroid_point - earth_point)
    return dist

def get_moid(a, e, w, i, om):
    # Returns Minimal Earth Orbit Intersection Distance
    # tmin = _get_rmin(a, e, w, i, om, direction="horizontal")
    ta0 = -w  # initial angular parameter in asteroid orbital frame
    te0 = om  # initial angular parameter in earth orbital frame
    tamin = so.fmin(partial(_find_dist, a=a, e=e, w=w, i=i, om=om), [ta0, te0], disp=False)
    # tamin = so.fmin(partial(_find_dist, a=a, e=e, w=w, i=i, om=om), [tmin, w], disp=False)
    moid = _find_dist(tamin, a, e, w, i, om)
    return moid 
    # return 0.05 + 0.02*(random.random() - random.random())

def _get_nonuniform_angles(n=100):
    # theta = np.linspace(0, 2*pi, numpoints)

    theta = []
    delta = pi/n
    t = 0
    base = 0
    # add, base = 0
    for p in range(n+1):
        angle = abs(base - pi*sin(t)) + base
        t += delta
        theta.append(angle)
        if p == ceil(n/2):
            base = pi

    theta = np.asarray(theta)
    # angles = [pi*p/float(numpoints) for p in range(numpoints)]
    # theta1 = [2*pi*sin(pi*an/float(numpoints)) for an in range(numpoints)]

    # logspace_pi = np.logspace(0.01, 1, int(numpoints*0.5))*0.1
    # lsp1 = (1 - logspace_pi)
    # sp1 = np.sort(pi*(lsp1 - lsp1[0] + 1))
    # lsp2 = (logspace_pi-logspace_pi[0])
    # sp2 = lsp2*pi/lsp2[-1] + pi
    # theta = np.concatenate((sp1,sp2[1:]))
    # print "theta:", theta
    # print "theta1:", theta1
    # print
    return theta


def get_orbpoints_hc(a, e, w, i, om, numpoints=100):
    theta = _get_nonuniform_angles(n=numpoints)
    points = np.array([_get_orbpoint(a, e, t) for t in theta])
    axis_w = np.array([cos(-w), sin(-w), 0.0])
    rotw = _get_rotmatrix(axis_w, i)
    points_inc = np.array([np.dot(point, rotw) for point in points])
    wb = om + w
    rotz = _get_rotmatrix(Z_AXIS, wb)
    points_hc = np.array([np.dot(point, rotz) for point in points_inc])
    return points_hc







# def get_incpoints(w, i, points):
#     # r = _get_r(a, e, w)
#     # axis = np.array([r*cos(w), r*sin(w), 0])
#     # axis0 = axis/norm(axis)
#     axis_w = np.array([cos(-w), sin(-w), 0.0])
#     rot = _get_rotmatrix(axis_w, i)
#     inc_points = []
#     for point in points:
#         inc_point = np.dot(point, rot)
#         inc_points.append(inc_point)
#     return np.asarray(inc_points)

# def _get_rotpoints(w, i, omega, inc_points):
#     angle = omega + w
#     zaxis = np.array([0.0, 0.0, 1.0])
#     rot = _get_rotmatrix(zaxis, angle)
#     rot_points = []
#     for point in inc_points:
#         rot_point = np.dot(point, rot)
#         rot_points.append(rot_point)
#     return np.asarray(rot_points)



# def _get_rmin(a, e, w, i, omega, direction):
#     if direction == 'horizontal':
#         func_min = so.fmin(partial(_get_rhor, a=a, e=e, w=w, i=i, omega=omega), pi/2, disp=False)
#     elif direction == 'vertical':
#         func_min = so.fmin(partial(_get_rver, a=a, e=e, w=w, i=i, omega=omega), pi/2, disp=False)
#     # func_min = so.fmin(func, pi/2)
#     return func_min

# def _get_rhor(t, a, e, w, i, om):   
#     rotpoint = rotated_orb_point(a, e, w, i, omega, t) 
#     # rotpoint = inclined_orb_point(a, e, w, i, t)
#     hpoint = rotpoint[:2]
#     r_hor = norm(hpoint)
#     return r_hor

# def _get_rver(t, a, e, w, i, omega):   
#     rotpoint = rotated_orb_point(a, e, w, i, omega, t) 
#     # rotpoint = inclined_orb_point(a, e, w, i, t)
#     # vpoint = rotpoint[1:]
#     # r_ver = norm(vpoint)
#     r_ver = np.dot(rotpoint, np.array([0.0, 0.0, 1.0]))
#     return r_ver


# def rotated_orb_point(a, e, w, i, omega, t):
#     inc_point = inclined_orb_point(a, e, w, i, t)
#     angle = omega + w
#     zaxis = np.array([0.0, 0.0, 1.0])
#     rotpoint = _rotate(inc_point, zaxis, angle)
#     return rotpoint

# def inclined_orb_point(a, e, w, i, t):
#     point = _get_orbpoint(a, e, t)
#     # r = _get_r(a, e, w)
#     # axis = np.array([r*cos(w), r*sin(w), 0])
#     # axis0 = axis/norm(axis)
#     axis0 = np.array([cos(-w), sin(-w), 0])
#     rotpoint = _rotate(point, axis0, i)
#     return rotpoint


# def _get_rxry(a, e, w, i, omega):
#     tmin = _get_rmin(a, e, w, i, omega, direction="horizontal")
#     rx = _get_rhor(tmin, a, e, w, i, omega)
#     # tmin = _get_rmin(a, e, w, i, omega, direction="vertical")
#     ry = _get_rver(tmin, a, e, w, i, omega)
#     return rx, ry

# # def _get_rw(a, e, w, i, omega):
# #     tmin = _get_rmin(a, e, w, i, omega, direction="horizontal")
# #     rx = _get_rhor(tmin, a, e, w, i, omega)
# #     # tmin = _get_rmin(a, e, w, i, omega, direction="vertical")
# #     ry = _get_rver(tmin, a, e, w, i, omega)
# #     return rx, ry


# # def get_earthorbpoint_hc(alpha):
# #     x = cos(alpha)
# #     y = sin(alpha)
# #     return [x, y, 0]



# # def _get_rmin(a, e, w, i, omega, direction):
# #     if direction == 'horizontal':
# #         func = partial(_get_rhor, a=a, e=e, w=w, i=i, omega=omega)
# #     elif direction == 'vertical':
# #         func = partial(_get_rver, a=a, e=e, w=w, i=i, omega=omega)
# #     func_min = so.fmin(func, pi/2)
# #     return func_min



# def find_center(a, e, w, i, omega):
#     c = np.array([-1,0,0])*(a*e)
#     # r = _get_r(a, e, w)
#     # axis = np.array([r*cos(w), r*sin(w), 0])
#     # axis0 = axis/norm(axis)
#     axis0 = np.array([cos(w), sin(w), 0])
#     c_inc = _rotate(c, axis0, i)
#     angle = omega + w
#     zaxis = np.array([0.0, 0.0, 1.0])
#     c_rot = _rotate(c_inc, zaxis, angle)
#     return c_rot


# def project_center(c):
#     cxy = norm(c[:2])

#     # cz = norm(c[1:])
#     cz = np.dot(c, np.array([0,0,1]))
#     return cxy, cz

# def _get_rxrycxcy(a, e, w, i, omega):
#     tmin = _get_rmin(a, e, w, i, omega, direction="horizontal")
#     rx = _get_rhor(tmin, a, e, w, i, omega)
#     # tmin = _get_rmin(a, e, w, i, omega, direction="vertical")
#     ry = _get_rver(tmin, a, e, w, i, omega)
#     c = find_center(a, e, w, i, omega)
#     cx, cy = project_center(c)
#     return rx, ry, cx, cy







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


    # tmin = _get_rmin(a, e, w, i, om, direction="horizontal")

    # rx = _get_rhor(tmin, a, e, w, i, om)
    # ry = _get_rver(tmin, a, e, w, i, om)
    rx, ry = _get_rxry(a, e, w, i, om)
    print "rx, ry:", rx, ry

    moid = get_moid(a, e, w, i, om)
    print "moid:", moid


    # vmin = _get_rmin(a, e, w, i, om, direction="vertical")
    # print "hmin, vmin:", hmin, vmin
    # print "_get_rhor(0.086603, a, e, w, i, om):", _get_rhor(0.086603, a, e, w, i, om)
    # print "_get_rhor(0.0, a, e, w, i, om):", _get_rhor(0.0, a, e, w, i, om)
    # print _get_r1(0.186585)

    orb_points = get_points(a, e)
    inc_points = get_incpoints(w, i, orb_points)
    rot_points = _get_rotpoints(w, i, om, inc_points)

    theta = np.linspace(0, 2*pi, 100)
    earthpoints = np.array([get_earthorbpoint_hc(t) for t in theta])

    # plot_orbits2D(orb_points, inc_points, rot_points)
    plot_orbits2D(orb_points, earthpoints, rot_points)

    # plt.plot(points, color="red")
    # plt.plot(points_rot, color="blue")
    # plt.grid(True)
    # plt.show()
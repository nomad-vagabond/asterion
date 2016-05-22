# -*- coding: utf-8 -*-
from OpenGL import GL, GLU, GLUT
import sys
import numpy as np
from math import pi

import calculate_orbits as co
from read_database import loadObject

class OrbitDisplayGL(object):

    def __init__(self, hazdata=None, nohazdata=None, mode='orbit'):
        # self.data = data
        self.mouseChangeX=0
        self.mouseChangeY=0
        self.angleX = -70.0
        self.angleY = -135.0
        self.hazdata = hazdata
        self.nohazdata = nohazdata
        self.mode = mode

    def prepare_displists(self):
        # self.create_triangles()
        # self.displists = [1]
        self.construct_earthorbit()
        self.create_earthorblist(1)
        if self.hazdata is not None:
            if self.mode == 'orbit':
                flat, inclined, self.haz_orbits = self.construct_orbits(self.hazdata)
                # self.haz_orbits, inclined, rotated = self.construct_orbits(self.hazdata)
                # flat, self.haz_orbits, rotated = self.construct_orbits(self.hazdata)
                self.create_orbitlist(2, self.haz_orbits)
            elif self.mode == 'point':
                self.construct_pointcloud(2, self.hazdata)
        if self.nohazdata is not None:
            if self.mode == 'orbit':
                flat, inclined, self.nohaz_orbits = self.construct_orbits(self.nohazdata)
                # self.nohaz_orbits, inclined, rotated = self.construct_orbits(self.nohazdata)
                # flat, self.nohaz_orbits, rotated = self.construct_orbits(self.nohazdata)
                self.create_orbitlist(3, self.nohaz_orbits)
            elif self.mode == 'point':
                self.construct_pointcloud(3, self.nohazdata)

    def InitGL(self, Width, Height):
        GL.glClearColor(0.1, 0.1, 0.1, 0.0)
        GL.glClearDepth(1.0)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glEnable(GL.GL_DEPTH_TEST)

        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_LIGHT1)
        # GL.glLightfv(GL.GL_LIGHT1, GL.GL_DIFFUSE, (0.2, 0.3, 0.5, 1.0))
        GL.glEnable(GL.GL_NORMALIZE)

        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(45.0, float(Width)/float(Height), 0.1, 100.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def ReSizeGLScene(self, Width, Height):
        if Height == 0: 
            Height = 1
        GL.glViewport(0, 0, Width, Height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(45.0, float(Width)/float(Height), 0.1, 100.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def DrawGLScene(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glLoadIdentity()
        GL.glTranslatef(0.0, 0.0, -10.0)
        GL.glRotatef(self.angleX, 1.0, 0.0, 0.0)
        GL.glRotatef(self.angleY, 0.0, 0.0, 1.0)
        GL.glEnable(GL.GL_ALPHA_TEST)
        # GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        # GL.glRotatef(rot, 0.0, 0.0, 1.0)


        if self.hazdata is not None:
            # GL.glDepthMask(GL.GL_FALSE)
            GL.glColor4f(0.9, 0.2, 0.2, 0.65)
            GL.glLineWidth(2)
            GL.glDisable(GL.GL_LIGHTING)
            GL.glCallList(2)
            GL.glEnable(GL.GL_LIGHTING)
            # GL.glDepthMask(GL.GL_TRUE)


        if self.nohazdata is not None:
            # GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            # GL.glEnable(GL.GL_BLEND)
            # GL.glDepthMask(GL.GL_FALSE)
            # GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glColor4f(0.0, 0.6, 0.9, 0.5)
            GL.glLineWidth(0.5)
            GL.glDisable(GL.GL_LIGHTING)
            GL.glCallList(3)
            GL.glEnable(GL.GL_LIGHTING)
            # GL.glDepthMask(GL.GL_TRUE)
            # GL.glDisable(GL.GL_BLEND)
            # GL.glEnable(GL.GL_DEPTH_TEST)

        # GL.glDepthMask(GL.GL_FALSE)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glColor4f(0.0, 0.7, 0.1, 1)
        GL.glLineWidth(4)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glCallList(1)
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_DEPTH_TEST)
        # GL.glDepthMask(GL.GL_TRUE)
        # GL.glEnable(GL.GL_BLEND)
        GLUT.glutSwapBuffers()

    def mouseHandle(self, button, state, x, y):
        if (button == GLUT.GLUT_LEFT_BUTTON) and (state == GLUT.GLUT_DOWN):
            self.mouseChangeX = x
            self.mouseChangeY = y

    def motionFunc(self, x,y):
        self.angleX -= (self.mouseChangeY-y)*0.1
        self.mouseChangeY = y
        self.angleY -= (self.mouseChangeX-x)*0.1
        self.mouseChangeX = x
        GLUT.glutPostRedisplay()

    def KeyPressed(self, *args):
        if args[0]=='\033': sys.exit()

    def create_orbitlist(self, nlist, orbits):
        GL.glNewList(nlist, GL.GL_COMPILE)
        GL.glBegin(GL.GL_LINES)
        for orbit in orbits:
            for i in range(len(orbit)-1):
                p1 = orbit[i]
                p2 = orbit[i+1]
                GL.glVertex3f(p1[0], p1[1], p1[2])
                GL.glVertex3f(p2[0], p2[1], p2[2])
        GL.glEnd()
        GL.glEndList()

    def create_earthorblist(self, nlist):
        GL.glNewList(nlist, GL.GL_COMPILE)
        GL.glBegin(GL.GL_LINES)
        for i in range(len(self.earthpoints)-1):
            p1 = self.earthpoints[i]
            p2 = self.earthpoints[i+1]
            GL.glVertex3f(p1[0], p1[1], p1[2])
            GL.glVertex3f(p2[0], p2[1], p2[2])
        GL.glEnd()
        GL.glEndList()
        
    def construct_orbits(self, data):
        flatorbits = []
        incorbits = []
        rotorbits = []
        for row in data:
            a, e, i, w, omega = row
            w, i, omega = np.radians([w, i, omega])
            flat_points = co.get_points(a, e, w, i, numpoints=30)
            inc_points = co.get_incpoints(a, e, w, i, flat_points)
            rot_points = co.get_rotpoints(a, e, w, i, omega, inc_points)
            flatorbits.append(flat_points)
            incorbits.append(inc_points)
            rotorbits.append(rot_points)
        return flatorbits, incorbits, rotorbits

    def construct_earthorbit(self):
        theta = np.linspace(0, 2*pi, 100)
        self.earthpoints = np.array([co.get_earthorb_point(t) for t in theta])

    def construct_pointcloud(self, nlist, pointdata):
        # GL.glEnable(GL.GL_POINT_SMOOTH)
        GL.glNewList(nlist, GL.GL_COMPILE)
        GL.glPointSize(3)
        GL.glBegin(GL.GL_POINTS)
        # i = 0
        for point in pointdata:
            # if i == 0:
            #     print "point:", point
            GL.glVertex3f(point[0], point[1], point[2])
        GL.glEnd()
        GL.glEndList()

    def show(self):

        rot = 0
        GLUT.glutInit(sys.argv)
        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | 
        GLUT.GLUT_ALPHA | GLUT.GLUT_DEPTH)
        GLUT.glutInitWindowSize(400, 300)
        GLUT.glutInitWindowPosition(0, 0)
        GLUT.glutCreateWindow('Asteroid Orbits')
        self.prepare_displists()
        GLUT.glutDisplayFunc(self.DrawGLScene)
        # GLUT.glutDisplayFunc(self.display_orbits)
        GLUT.glutMouseFunc(self.mouseHandle)
        GLUT.glutMotionFunc(self.motionFunc)
        # GLUT.glutIdleFunc(DrawGLScene)
        GLUT.glutReshapeFunc(self.ReSizeGLScene)
        GLUT.glutKeyboardFunc(self.KeyPressed)
        self.InitGL(400, 300)
        GLUT.glutMainLoop()


if __name__ == '__main__':

    sources = ['./asteroid_data/haz_rand_small.p',
               './asteroid_data/nohaz_rand_small.p']

    # sources = ['./asteroid_data/haz.p', './asteroid_data/nohaz.p']
    cutcol = ['a', 'e', 'i', 'w', 'om']
    # cutdata = dataset[cutcol]
    datasets = map(loadObject, sources)
    hazdata_gen, nohazdata_gen = [dataset[cutcol].as_matrix() for dataset in datasets]
    # print "hazdata_gen:", hazdata_gen[:5]
    disp_orbit = OrbitDisplayGL(hazdata=hazdata_gen, 
                                nohazdata=nohazdata_gen, mode='orbit')
    disp_orbit.show()

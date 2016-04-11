#!/usr/bin/env python

import numpy
import os
import signal
import sys
import cv2

"""Module to implement a basic point cloud viewer using OpenGL or OpenCV."""

try:
    import OpenGL.GL as gl
    import OpenGL.GLU as glu
    import OpenGL.GLUT as glut
    _RENDER_METHOD = 'opengl'
except:
    _RENDER_METHOD = 'opencv'

_FOVY = numpy.pi/4
_F_NORMALIZED = 0.5 / numpy.tan(0.5*_FOVY)

######################################################################

def normalize_data(xyz):

    # re-mean to zero
    #xyz_mean = 0.5 * (xyz.max(axis=0) + xyz.min(axis=0))
    xyz_mean = xyz.mean(axis=0)
    xyz -= xyz_mean

    y = numpy.abs(xyz[:,1])
    z = xyz[:,2]

    k = 2*_F_NORMALIZED*y - z

    z += k.max()

######################################################################
    
def gradient(U):

    '''Prety rainbow colors for display'''

    T = U.dtype

    colors = numpy.array([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        ], dtype=T)

    stops = numpy.arange(colors.shape[0], dtype=T)/(colors.shape[0]-1)

    rgbs = [ numpy.interp(U, stops, colors[:,i]).reshape(-1,1) 
             for i in range(3) ]

    return numpy.hstack( rgbs )

######################################################################

class PointCloudAppOpenCV:

    """Class to implement a basic point cloud viewer using OpenCV."""

    def __init__(self, xyz, color=None):

        # Check shape
        if ( len(xyz.shape) != 2 or xyz.shape[1] != 3 ):
            raise Exception('xyz must be an n-by-3 array')

        
        # Store points
        self.npoints = xyz.shape[0]

        self.xyz = xyz.astype('float32')
        normalize_data(self.xyz)

        # Compute mean Z value and range
        Z = self.xyz[:,2]
        self.zmean = numpy.average(Z)

        zmin = Z.min()
        zmax = Z.max()

        if zmax > zmin:
            zrng = zmax-zmin
        else:
            zrng = 1

        # Compute colors for points
        if color is None:
            self.rgb = gradient((Z - zmin)/zrng)
            self.rgb = (self.rgb*255).astype('uint8')
        else:
            self.rgb = color

        num_desired = 640*480
        idx = numpy.arange(self.npoints)

        if self.npoints > num_desired:
            numpy.random.shuffle(idx)
            idx = idx[:num_desired]

        self.draw_points = self.xyz[ idx, : ].transpose()
        nidx = len(idx)

        self.draw_points = numpy.vstack(
            ( self.draw_points,
              numpy.ones((1,nidx),dtype='float32') ) )

        self.draw_rgb = self.rgb[ idx, : ]
        self.draw_rgb = self.draw_rgb[ :, [2,1,0] ]

        # Set up X/Y rotations and mouse state
        self.rot = numpy.array([0,0], dtype='float32')
        self.lastMouse = None
        
        self.big_points = False
        
        self.dragging = False
        self.need_redisplay = True
        self.mouse_point = None
        
        self.win = 'Point cloud'
        cv2.namedWindow(self.win)

        print 'Press space to reset view, P to toggle point size, ESC to quit.'


    def mouseEvent(self, event, x, y, flags, param):

        p = numpy.array([x,y], dtype='float32')

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.mouse_point = p
        elif self.dragging and event == cv2.EVENT_MOUSEMOVE:
            diff = p - self.mouse_point
            self.mouse_point = p
            self.rot += diff
            self.need_redisplay = True
        elif self.dragging and event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.mouse_point = None

    def run(self):

        self.need_redisplay = True

        cv2.setMouseCallback(self.win, self.mouseEvent, self)

        while True:

            if self.need_redisplay:
                self.display()
                self.need_redisplay = False

            k = cv2.waitKey(5)

            if k >= 32 and k < 127:
                c = chr(k).lower()
                if c == 'p':
                    self.big_points = not self.big_points
                    self.need_redisplay = True
                elif c == ' ':
                    self.rot[:] = 0
                    self.need_redisplay = True
            elif k == 27:
                break

        cv2.setMouseCallback(self.win, lambda e,x,y,f,p: None, None)

                    
    def getModelview(self):

        ry, rx = tuple(self.rot*numpy.pi/180)

        cx = numpy.cos(rx)
        sx = numpy.sin(rx)

        cy = numpy.cos(ry)
        sy = numpy.sin(ry)

        Rx = numpy.array( [ [ 1,  0,   0, 0 ],
                            [ 0, cx, -sx, 0 ],
                            [ 0, sx,  cx, 0 ],
                            [ 0,  0,   0, 1 ], ], dtype='float32' )

        Ry = numpy.array( [ [  cy, 0, -sy, 0 ],
                            [   0, 1,   0, 0 ],
                            [  sy, 0,  cy, 0 ],
                            [   0, 0,   0, 1 ] ], dtype='float32' )
        
        M = numpy.array( [ 
            [ 1.0,  0.0,  0.0, 0.0 ],
            [ 0.0, -1.0,  0.0, 0.0 ],
            [ 0.0,  0.0, -1.0, 0.0 ],
            [ 0.0,  0.0,  0.0, 1.0 ] ], dtype='float32')

        T1 = numpy.eye(4, dtype='float32')
        T2 = numpy.eye(4, dtype='float32')

        T1[2,3] = self.zmean
        T2[2,3] = -self.zmean

        M = numpy.dot(M, T1)
        M = numpy.dot(M, Ry)
        M = numpy.dot(M, Rx)
        M = numpy.dot(M, T2)

        return M

    def display(self):

        w = 640
        h = 480
        f = h * _F_NORMALIZED

        display = numpy.zeros((h*w, 3), dtype='uint8')

        M = self.getModelview()

        points2 = numpy.dot(M, self.draw_points)

        x = points2[0,:]
        y = points2[1,:]
        z = points2[2,:]

        # Perspective divide (note: I have no clue why x is negated
        # here - MZ)
        u = ( (f * -x/z) + w/2 + 0.5).astype(int)
        v = ( (f * y/z) + h/2 + 0.5).astype(int)
        color_idx = numpy.arange( 0, len(u) )
        
        # Add neighbors
        if self.big_points:
            u = numpy.hstack( (u+0, u+1, u+0, u+1) )
            v = numpy.hstack( (v+0, v+0, v+1, v+1) )
            z = numpy.tile( z, 4 )
            color_idx = numpy.tile( color_idx, 4 )

        # Viewport culling
        mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        pixel_idx = u[mask] + v[mask]*w
        color_idx = color_idx[mask]

        # Depth sorting
        depth = z[mask]
        depth_idx = numpy.arange(0, len(depth))
        depth_idx = numpy.argsort(depth, axis=None)

        pixel_idx = pixel_idx[depth_idx]
        color_idx = color_idx[depth_idx]

        # Splat!
        display[pixel_idx,:] = self.draw_rgb[color_idx]

        cv2.imshow(self.win, display.reshape((h,w,3)))

        

######################################################################

    

class PointCloudAppOpenGL:

    """Class to implement a basic point cloud viewer using OpenGL."""

    def __init__(self, xyz, color=None):

        """Initialize a PointCloudApp with the given XYZ data. The xyz
        parameter should be an N-by-3 numpy array of data to be
        visualized. If the data type of the array is not float32, then
        it will be converted automatically.

        """


        # Check shape
        if ( len(xyz.shape) != 2 or xyz.shape[1] != 3 ):
            raise Exception('xyz must be an n-by-3 array')

        # Store points
        self.npoints = xyz.shape[0]

        self.xyz = xyz.astype('float32')
        normalize_data(self.xyz)

        # Compute mean Z value and range
        Z = self.xyz[:,2]
        self.zmean = numpy.average(Z)

        zmin = Z.min()
        zmax = Z.max()

        if zmax > zmin:
            zrng = zmax-zmin
        else:
            zrng = 1

        # Compute color for points
        if color is None:
            self.rgb = gradient((Z - zmin)/zrng)
        else:
            self.rgb = color.astype(float)/255.0


        # Set up GLUT
        glut.glutInit()
        glut.glutInitWindowSize(640, 480)
        glut.glutInitDisplayMode( glut.GLUT_DOUBLE | glut.GLUT_RGB | 
                                  glut.GLUT_DEPTH | glut.GLUT_MULTISAMPLE )
        glut.glutCreateWindow('Point cloud viewer')
        glut.glutDisplayFunc(self.display)
        glut.glutReshapeFunc(self.reshape)
        glut.glutKeyboardFunc(self.keyboard)
        glut.glutMouseFunc(self.mouse)
        glut.glutMotionFunc(self.motion)

        # Matrix to create a coordinate system with 
        #  X right
        #  Y down
        #  Z forward
        self.M = numpy.array( [ 
            [ 1.0,  0.0,  0.0, 0.0 ],
            [ 0.0, -1.0,  0.0, 0.0 ],
            [ 0.0,  0.0, -1.0, 0.0 ],
            [ 0.0,  0.0,  0.0, 1.0 ] ] )


        # Set up X/Y rotations and mouse state
        self.rot = numpy.array([0,0], dtype='float32')
        self.lastMouse = None

        # Handle Ctrl-C gracefully
        signal.signal(signal.SIGINT, self._ctrlC)
        glut.glutTimerFunc(100, self._watchdog, 0)

        # Set up OpenGL state
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glPointSize(2.0)

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)

        gl.glVertexPointer(3, gl.GL_FLOAT, 0, self.xyz)
        gl.glColorPointer(3, gl.GL_FLOAT, 0, self.rgb)

        gl.glEnable(gl.GL_DEPTH_TEST)

        print 'Press space to reset view, ESC to quit.'

    def _watchdog(self, value):
        # Used only to handle Ctrl-C 
        glut.glutTimerFunc(100, self._watchdog, 0)

    def _ctrlC(self, signum, frame):
        # Used only to handle Ctrl-C 
        print 'Ctrl-C pressed, exiting.'
        sys.exit(0)

    def mouse(self, button, state, x, y):
        if button == glut.GLUT_LEFT_BUTTON: 
            # Left mouse button pressed/or released
            if state == glut.GLUT_DOWN:
                # LMB pressed, set last known position
                self.lastMouse = numpy.array([x,y])
            else:
                # LMB released, get rid of mouse position
                self.lastMouse = None

    def motion(self, x, y):
        if self.lastMouse is not None:
            # If mouse pressed, increment X/Y rotation by amount of mouse motion
            curMouse = numpy.array([x,y])
            diff = curMouse - self.lastMouse
            self.lastMouse = curMouse
            self.rot += diff
            glut.glutPostRedisplay()

    def display(self):

        # Clear the screen
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        # Set up the view matrix by rotating scene about mean z coordinate
        gl.glPushMatrix()
        gl.glTranslatef(0, 0, self.zmean)
        gl.glRotatef(-self.rot[0], 0, 1, 0)
        gl.glRotatef(self.rot[1], 1, 0, 0)
        gl.glTranslatef(0, 0, -self.zmean)

        # Draw the points
        gl.glDrawArrays(gl.GL_POINTS, 0, self.xyz.shape[0])

        # Restore view matrix
        gl.glPopMatrix()
        
        # Tell GLUT to draw!
        glut.glutSwapBuffers()

    def reshape(self, w, h):

        # Handle window resizing
        gl.glViewport(0,0,w,h)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()

        if not h:
            aspect = 1
        else:
            aspect = float(w)/h

        glu.gluPerspective(_FOVY * 180 / numpy.pi, aspect, 0.1, 100)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glMultMatrixd(self.M)
        
    def keyboard(self, key, x, y):
        # Keyboard pressed
        if (key == '\x1b'):
            print 'ESC pressed, exiting.'
            sys.exit(0)
        elif key == ' ':
            self.rot = numpy.array([0,0], dtype='float32')
            glut.glutPostRedisplay()

    def run(self):
        # Run this app
        glut.glutMainLoop()

######################################################################

def PointCloudApp(xyz, color=None, allow_opengl=True):
    if _RENDER_METHOD == 'opengl' and allow_opengl:
        return PointCloudAppOpenGL(xyz, color)
    else:
        print 'Using OpenCV for software rendering. This may be slow and crappy-looking.'
        print
        if _RENDER_METHOD != 'opengl':
            print 'Install the OpenGL package to get decent-looking, speedy output!'
            print
        return PointCloudAppOpenCV(xyz, color)

######################################################################

if __name__ == '__main__':

    allow_opengl = True

    if len(sys.argv) > 1 and sys.argv[1] == '-nogl':
        allow_opengl = False
        sys.argv.pop(1)
    elif _RENDER_METHOD == 'opengl':
        print 'You can run this program with -nogl as the first argument'
        print 'to force software rendering with OpenCV.'
        print

    if len(sys.argv) == 2:

        datafile = sys.argv[1]
        xyz = numpy.load(datafile)
        if datafile.lower().endswith('.npz'):
            xyz = xyz[xyz.keys()[0]]

    else:

        print 'You can also run this program using\n'
        print '  ' + os.path.basename(sys.argv[0]) + ' xyz.npy'
        print '  ' + os.path.basename(sys.argv[0]) + ' xyz.npz\n'
        print 'where the first argument is XYZ data in numpy npy or npz format.'
        print
        
        # make a nice happy torus 5 meters away from the camera

        # equal spacing for X/Y from -2 to 2 meters
        xyrange = numpy.linspace(-2, 2, 500).astype('float32')
        X,Y = numpy.meshgrid(xyrange, xyrange)

        # initialize an empty array for Z
        Z = numpy.empty_like(X)

        # Major and minor radii for torus
        R1 = 1.2
        R2 = 0.7

        # distance from each X/Y element to torus center
        d = numpy.sqrt(X**2 + Y**2)

        # find the X/Y locations which are inside the torus
        mask = numpy.logical_and(d >= R1-R2, d <= R1+R2)
        
        # compute the Z values for the torus
        Z[mask] = 5 - numpy.sqrt( R2**2 - (d[mask] - R1)**2 ) 
        
        # fill in an XYZ array for ONLY the valid XY locations
        xyz = numpy.hstack( ( X[mask].reshape((-1,1)),
                              Y[mask].reshape((-1,1)),
                              Z[mask].reshape((-1,1)) ) )

    app = PointCloudApp(xyz, allow_opengl=allow_opengl)

    
    app.run()

    

import numpy as np
import cv2
import sys
from PointCloudApp import PointCloudApp, _RENDER_METHOD

# check to see whether we should disable OpenGL rendering
allow_opengl = True

if len(sys.argv) > 1 and sys.argv[1] == '-nogl':
    allow_opengl = False
    sys.argv.pop(1)
elif _RENDER_METHOD == 'opengl':
    print 'You can run this program with -nogl as the first argument'
    print 'to force software rendering with OpenCV.'
    print

# check for command-line argument
if len(sys.argv) != 2:
    print 'usage:', sys.argv[0] + ' DATAFILE.npz'
    print
    sys.exit(0)

# load data file
data = np.load(sys.argv[1])

# grab rgb and depth images
xyz = data['xyz']
color = data['color']

# display it
pcl = PointCloudApp(xyz, color, allow_opengl)
pcl.run()

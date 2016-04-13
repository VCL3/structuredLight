#!/usr/bin/env python

import cv2
import numpy
import sys
import os
from PointCloudApp import PointCloudApp

# Get command line arguments or print usage and exit
if len(sys.argv) > 2:
    proj_file = sys.argv[1]
    cam_file = sys.argv[2]
else:
    progname = os.path.basename(sys.argv[0])
    print >> sys.stderr, 'usage: '+progname+' PROJIMAGE CAMIMAGE'
    sys.exit(1)

# Load in our images as grayscale (1 channel) images
proj_image = cv2.imread(proj_file, cv2.IMREAD_GRAYSCALE)
cam_image = cv2.imread(cam_file, cv2.IMREAD_GRAYSCALE)

# Make sure they are the same size.
assert(proj_image.shape == cam_image.shape)

# Set up parameters for stereo matching (see OpenCV docs at
# http://goo.gl/U5iW51 for details).
min_disparity = 0
max_disparity = 16
window_size = 35
param_P1 = 0
param_P2 = 20000

# Create a stereo matcher object
matcher = cv2.StereoSGBM_create(min_disparity, 
                                max_disparity, 
                                window_size, 
                                param_P1, 
                                param_P2)

# Compute a disparity image. The actual disparity image is in
# fixed-point format and needs to be divided by 16 to convert to
# actual disparities.
disparity = matcher.compute(cam_image, proj_image) / 16.0

# Pop up the disparity image.
cv2.imshow('Disparity', disparity/disparity.max())
while cv2.waitKey(5) < 0: pass

# Get the shape of disparity image, calculate n
w, h = disparity.shape
n = w * h

# Set up the parameters used to get the XYZ data
b = 0.05
f = 600

# Try to create a 3*N matrix where each column is a point
# In the format of [i,j,Z]T(ranspose)
x_range = numpy.arange(0,h,1)
y_range = numpy.arange(0,w,1)
xx, yy = numpy.meshgrid(x_range,y_range) 

# Create a mask by checking the disparity criteria
Z_max = 8
thres = b * f / Z_max
mask = numpy.greater(disparity,thres)

# Calculate the depths
Z = b * f / disparity

# Hardcode the inverse of calibration matrix
K_inv = numpy.matrix([[1/float(600),0,-8/float(15)],
                     [0,1/float(600),-2/float(5)],
                     [0,0,1]])

d = Z[mask].reshape((-1,1))

# For each (xx, yy), multiply them by its corresponding d
# Then stack them together to build V transpose
mat = numpy.hstack( ( numpy.multiply(xx[mask].reshape((-1,1)),d),
                      numpy.multiply(yy[mask].reshape((-1,1)),d),
                      d) )

# Finally use a simple matrix multiplication to find the n-by-3 point set
xyz = numpy.array(mat * K_inv.transpose())

"""
# Non-vectorized version
# For each point in disparity map construct a XYZ point
K_inv_ = numpy.array([[1/float(600),0,-8/float(15)],
                     [0,1/float(600),-2/float(5)],
                     [0,0,1]])
xyz = []
Z_max = 8
thres = b * f / Z_max
for i in range(w):
    for j in range(h):
        if disparity[i][j] > thres:
            Z = b * f / disparity[i][j]
            p_cam = numpy.dot(K_inv_, [i,j,1])
            p_cam *= Z 
            p_cam[0], p_cam[1] = p_cam[1], p_cam[0] 
            xyz.append(p_cam)
xyz = numpy.array(xyz)
"""

# Call point cloud to render
pcl = PointCloudApp(xyz, None, True)
pcl.run()

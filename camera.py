# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Convert BGR to GRAYSCALE
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def binary_threshold(img, thresh):
    """
    Create a binary mask
    --------------------
    Function to create a binary mask where thresholds are met.

    1. Apply the threshold, zeros otherwise.
    2. Return the binary image.
    """
    binary = np.zeros_like(img)
    binary[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return binary

class Camera():
  def __init__(self):
    self.M = None
    self.Minv = None
    self.mtx = None
    self.dist = None

    self.set_transform_matrix()
    if os.path.exists('mtx_dist.p'):
        self.load_calibration()
    else:
        self.calibrate(nx=9, ny=6)

  def set_transform_matrix(self):
    """
    Set the perspective Transform matrix and its inverse
    """
    src = np.float32([[240,719], [579,450], [712,450],[1165,719]])
    dst = np.float32([[300,719], [300,0], [900,0],[900,719]])
    self.M = cv2.getPerspectiveTransform(src, dst)
    self.Minv = cv2.getPerspectiveTransform(dst, src)

  def find_corners(self, img, nx, ny):
    """
    Function that takes an image, number of x and y points,
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, add object points, image points
    if found == True:
        img = cv2.drawChessboardCorners(img, (nx, ny), corners, found)
        plt.imshow(img);
    return found, corners

  def load_calibration(self):
    print('Loading camera calibration file...')
    dist_pickle = pickle.load(open("mtx_dist.p", "rb"))
    self.mtx = dist_pickle["mtx"]
    self.dist = dist_pickle["dist"]

  def calibrate(self, nx=9, ny=6):
    print('Calibrating camera...')
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Step through the list and search for chessboard corners
    filenames = glob.glob('camera_cal/calibration*.jpg')
    for filename in filenames:
        img = cv2.imread(filename)
        ret, corners = self.find_corners(img, nx, ny)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    img_size = (img.shape[1], img.shape[0])
    ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Save the calibration file for future uses
    dist_pickle = {}
    dist_pickle["mtx"] = self.mtx
    dist_pickle["dist"] = self.dist
    pickle.dump( dist_pickle, open( "mtx_dist.p", "wb" ) )


  def undistort_image(self, img):
    """
    Apply distortion correction to the image
    """
    undistorted = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
    return undistorted


  def mag_threshold(self, img, sobel_kernel=3, thresh=(0, 255)):
    """
    Gradient magnitude
    ------------------
    Function to threshold an image for a given gradient magnitude range.

    The magnitude, or absolute value of the gradient is just the square
    root of the squares of the individual x and y gradients;
    for a gradient in both the x and y directions, the magnitude
    is the square root of the sum of the squares.

    1. Take both Sobel x and y gradients.
    2. Calculate the gradient magnitude.
    3. Scale to 8-bit (0 - 255) and convert to type = np.uint8
    4. Return the binary thresholded image.
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    return binary_threshold(gradmag, thresh)


  def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Gradient direction threshold
    ----------------------------
    Function to threshold an image for a given range and Sobel kernel.

    The direction of the gradient is simply the arctangent of the y-gradient
    divided by the x-gradient. Each pixel of the resulting image contains a
    value for the angle of the gradient away from horizontal in units of radians,
    covering a range of −π/2 to π/2. An orientation of 0 implies a horizontal
    line and orientations of +/−π/2 imply vertical lines.

    1. Take both Sobel x and y gradients.
    2. Take the absolute value of the gradient direction.
    3. Return the binary thresholded image.
    """
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    return binary_threshold(absgraddir, thresh)


  def hls_threshold(self, img, thresh=(0, 255)):
    """
    Color space threshold
    ---------------------
    Function to threshold the S-channel of an image.

    1. Convert from RGB to HLS.
    2. Extract the S-Channel.
    4. Return the binary thresholded image.
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    return binary_threshold(s_channel, thresh)

  def multi_thresholds(self, img):
    """
    Combine thresholds
    """
    gray = grayscale(img)
    magnitude = self.mag_threshold(gray, sobel_kernel=9, thresh=(50, 155))
    direction = self.dir_threshold(gray, sobel_kernel=15, thresh=(0.7,1.4))
    combined_binary = np.zeros_like(direction)
    combined_binary[((magnitude == 1) & (direction == 1))] = 1
    hls = self.hls_threshold(img, thresh=(100, 255))
    combined = np.zeros_like(combined_binary)
    combined[(hls == 1) | (combined_binary == 1)] = 1
    return combined

  def warp(self, img):
    """
    Define a perspective transform function
    """
    warped = cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped

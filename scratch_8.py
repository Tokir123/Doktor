import cv2 as cv
import numpy as np
from PIL import Image
from CnM.generators import *
from CnM.models import *
from CnM.runtime_methods import *

import SimpleITK as sitk
import keras
import tensorflow as tf
import time

os.chdir("..")
image  = sitk.ReadImage('normmm/saveverzgood.mha')
image  = sitk.GetArrayFromImage(image)
kernel = np.ones((4,4),np.uint8)

image_1=image

for i in range(100):

    #image_1[i,...] = cv.erode(image[i,...],kernel,iterations = 2)
    image_1[i,...] = cv.morphologyEx(image[i,...], cv.MORPH_CLOSE, kernel)


image_2 = sitk.GetImageFromArray(image_1)
sitk.WriteImage(image_2, 'eroded.mha')


# noise removal
thresh=image[50,...]
img=thresh
gray=thresh
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)
# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv.watershed(img,markers)

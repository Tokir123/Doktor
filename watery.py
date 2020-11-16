import numpy as np
from PIL import Image
import SimpleITK as sitk
import time
import cv2 as cv
from CnM.cust_callback import *
import copy
from CnM import gui

os.chdir("..")
print(os.getcwd())
####
image  = sitk.ReadImage('training_data/image/image.mha')
image = np.moveaxis(sitk.GetArrayFromImage(image), 0, -1)[106:602,320:800,2:98]
####

####
segmented  = sitk.ReadImage('inference/predictions/0.23534186215655029.mha')
segmented = np.moveaxis(sitk.GetArrayFromImage(segmented), 0, -1)
####
correcting = np.zeros(shape=segmented.shape)
correcting[30:correcting.shape[0] - 30, 30:correcting.shape[1] - 30, 10:correcting.shape[2] - 10] = 1
segmented=segmented*correcting

#sitk.connectedComponent(segmented)
#cv.connectedComponents(segmented)


img = sitk.ReadImage('training_data/image/image.mha')
gui.MultiImageDisplay(image_list = [img], figure_size=(8,4));
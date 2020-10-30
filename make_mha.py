import numpy as np
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk



file_list=os.listdir(label_folder)

im = cv2.imread(file_list[0], -1)
print(im.shape)
ph=np.zeros(shape=full_image_size+(len(file_list),))#+4

for i in range(len(file_list)):
    ph[...,i]=cv2.imread(file_list[0], -1)


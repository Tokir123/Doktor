import cv2
from PIL import Image
import os
import numpy as np
os.chdir("..") #Moves up one level

labels = np.array(Image.open('images/training/source/rotated/labels/7.png'))
cv2.connectedComponents(labels)
#Import Bild als image
#
#D:\PycharmProjects\Unet3D\images\training\source\rotated\labels
#

ret,mask=cv2.connectedComponents(labels,connectivity=8)
mask=mask-1
ret=ret-1

def max_neighbours(image,pixel):
    class_indication=0

    if not((min(pixel)<1)|(pixel[0]>image.shape[0]-1)|(pixel[1]>image.shape[1]-1)):
        class_indication=np.max(image[pixel[0]-1:pixel[0]+1,pixel[1]-1:pixel[1]+1])

    return class_indication





weighting_all=np.zeros(labels.shape+(ret,))

for i in range(ret):
    weighting_all[...,i]=(mask==i)

for iteration in range(10):
    for i in range(ret):
        print(i)
        for pixel in np.ndindex(weighting_all[...,i].shape):
             if (weighting_all[pixel+(i,)]==0):
                if(max_neighbours(weighting_all[...,i], pixel)>0):
                    weighting_all[pixel+(i,)] = max_neighbours(weighting_all[..., i], pixel)+1

weighting_all[weighting_all==0]=11
weighting_all=weighting_all-1

weighting=np.zeros(labels.shape)

for pixel in np.ndindex(labels.shape):
    if(labels[pixel]==0):
        weighting[pixel]=weight_function(np.unique(weighting_all[pixel,])[0],np.unique(weighting_all[pixel,])[1])
weighting=weighting+1


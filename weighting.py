import cv2
from PIL import Image
import os
import numpy as np
import math as m
import copy
os.chdir("..") #Moves up one level

labels = np.array(Image.open('training_data/labels/28.png'))
labels=labels[...,0]

#Import Bild als image
#
#D:\PycharmProjects\Unet3D\images\training\source\rotated\labels
#

ret,mask=cv2.connectedComponents(labels,connectivity=8)
mask=mask-1
ret=ret-1
print(ret)
def max_neighbours(image,pixel):
    class_indication=0

    if not((min(pixel)<1)|(pixel[0]>image.shape[0]-1)|(pixel[1]>image.shape[1]-1)):
        class_indication=np.max(image[pixel[0]-1:pixel[0]+2,pixel[1]-1:pixel[1]+2])

    return class_indication

def min_neighbours(image,pixel):
    class_indication=0

    if not((min(pixel)<1)|(pixel[0]>image.shape[0]-1)|(pixel[1]>image.shape[1]-1)):
        class_indication=np.unique(image[pixel[0]-1:pixel[0]+1,pixel[1]-1:pixel[1]+1])[1]

    return class_indication


def weight_function(value_1,value_2,weight_c=0,weight_m=1,max_distance=6):
    return_value=weight_c+weight_m*m.exp((-1)*(value_1**2+value_2**2)/(max_distance**2))
    return return_value

weighting_all=np.zeros(labels.shape+(ret,))

for i in range(ret):
    weighting_all[...,i]=(mask==i)

for iteration in range(5):
    snapshot=copy.deepcopy(weighting_all[...])
    print(np.unique(snapshot))
    print(snapshot is weighting_all)
    for i in range(ret):
        print(i)
        for pixel in np.ndindex(weighting_all[...,i].shape):
             if (weighting_all[pixel+(i,)]==0):
                 if(max_neighbours(snapshot[...,i], pixel)>0):
                     #weighting_all[pixel+(i,)] = min_neighbours(snapshot[..., i], pixel)+1
                     weighting_all[pixel + (i,)]=iteration+2

weighting_all[weighting_all==0]=9
weighting_all=weighting_all-1

weighting=np.zeros(labels.shape)



for pixel in np.ndindex(labels.shape):
    if(labels[pixel]==0):
        weighting[pixel]=weight_function(np.sort(weighting_all[pixel])[0],np.sort(weighting_all[pixel])[1])
weighting=(weighting/(np.max(weighting))*249).astype(np.uint8)
img = Image.fromarray(weighting)
img.show()

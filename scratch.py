import numpy as np
from PIL import Image
from CnM.generators import *
from CnM.models import *
import SimpleITK as sitk

print('d')
image = sitk.ReadImage('images/training/source/image.mha')
print('d')
image = np.moveaxis(sitk.GetArrayFromImage(image),0,-1)
#image=np.array(image)
labels=np.array(Image.open('images/training/source/labels.png'))
print(labels.shape)
dims=image.shape
print(dims)

image=image/(np.max(image))
image[image==0]=np.mean(image)

new_labels=np.zeros(shape=labels.shape+(3,))
for i in range(3):
    new_labels[...,i]= labels==i

onehot=new_labels

new_labels=np.zeros(image.shape+(3,))+4
new_labels[:,:,50,:]=onehot
labels=new_labels
image=image[...,np.newaxis]

min=np.array((0,0,0),dtype=int)
max=np.array((np.floor(0.75*dims[0]),dims[1],dims[2]),dtype=int)
size=80

dataGenerator=dataGen(image, labels, lower=min, upper=max, target_size=80, batch_size=8)

l=dataGenerator.__next__()[0]


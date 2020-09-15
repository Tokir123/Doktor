import numpy as np
from PIL import Image
from CnM.generators import *
from CnM.models import *
from CnM.runtime_methods import *
#from CnM.paral import *
import SimpleITK as sitk
import keras
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import keras.backend as K
from keras.models import load_model

#export TF_ENABLE_AUTO_MIXED_PRECISION=1

#K.set_floatx('float32')

####Import
image = sitk.ReadImage('images/training/source/image.mha')
labels=np.array(Image.open('images/training/source/labelsddd.png'))
####Import Done

image = np.moveaxis(sitk.GetArrayFromImage(image),0,-1)

image=image/(np.max(image))
image[image==0]=np.mean(image)
dims=image.shape
labels=MakeLabel(labels)

image=image[...,np.newaxis]


#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)

############
size=64
depth=64
class_weights=(0.5,1.2,100)
batch_size=1
downscale_factor=8
########






min=(0,0,0)
max=dims
target_size=(size,size,depth)
model=load_model('path_to_my_modedwl.h5')

#if __name__== "__main__":

image_pad=np.zeros(shape=(1204,1204,1204,1))
image_pad[80:1088,:,500:601,[0]]=image

for i in range(30):
    another_TTA(1, target_size, image_pad, model, 'try8')

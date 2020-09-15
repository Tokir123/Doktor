import numpy as np
from PIL import Image
from CnM.generators import *
from CnM.models import *
from CnM.runtime_methods import *

import SimpleITK as sitk
import keras
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import keras.backend as K
from keras.models import load_model

#export TF_ENABLE_AUTO_MIXED_PRECISION=1

#K.set_floatx('float32')

####Import
image = sitk.ReadImage('images/training/source/rotated/image.mha')
labels=np.array(Image.open('images/training/source/rotated/labelsd.png'))
####Import Done

image = np.moveaxis(sitk.GetArrayFromImage(image),0,-1)
mean=np.mean(image)
std=np.std(image)
image=(image-mean)/std


#image=image/(np.max(image))
#image[image==0]=np.mean(image)

dims=image.shape
labels=MakeLabel(full_image_size=image.shape)

image=image[...,np.newaxis]


#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)

############
size=80
depth=80
class_weights=(1,1,3)
batch_size=4
downscale_factor=2
folder_name='norm'
label_folder='adw'
########






min=(0,0,0)
max=dims
target_size=(size,size,depth)


dataGenerator=dataGen(image, labels, lower=min, upper=max, target_size=target_size, batch_size=batch_size*downscale_factor, slice_label=50,class_weights=class_weights)

myGen=transGenerator(dataGenerator)

model=unet3D(downscale_factor=downscale_factor,kernelSize=3,input_size=target_size+(1,),outputSize=3,activation='softmax',loss='categorical_crossentropy')
model=addWeightTo3DModel(model, keras.losses.categorical_crossentropy)
#model.load_weights(folder_name+'/my_model_weights.h5')
l=model.fit_generator(myGen,steps_per_epoch=7000,epochs=1)
#model.save(folder_name+'/model.h5')
model.save_weights(folder_name+'/my_model_weights.h5')


padded_image=ImagePadSym(image[...,0])
#big=Apply(ModelTo3D_single,model,data=padded_image[np.newaxis,...],input_size=target_size,padding=(16,16,8))
'''

big=Apply(ModelTo3D_single,model,data=image[np.newaxis,...,0],input_size=target_size,padding=(16,16,8))
#ModelTo3D_single(model,image)
#TTAHelper
bigg=big*120
bigg=bigg.astype(np.uint8)


bigg=np.swapaxes(bigg,0,-1)
bigg=np.swapaxes(bigg,1,2)
img=sitk.GetImageFromArray(bigg)
sitk.WriteImage(img, folder_name+'/big.mha')

#exit()
del bigg
del big
del img
del l

image_pad=np.zeros(shape=(1204,1204,1204,1))
image_pad[80:1088,:,500:601,[0]]=image
del image
del labels

for i in range(50):
    another_TTA(1, target_size, image_pad, model, folder_name+'/'+str(i)+'.mha')

'''
####Import
image = sitk.ReadImage('images/training/source/wda/image_.mha')

####Import Done

image = np.moveaxis(sitk.GetArrayFromImage(image),0,-1)
for i in range(9):
    image[...,i]=image[...,9]
mean=np.mean(image[image!=0])
std=np.std(image[image!=0])
image_norm=image
image_norm[image==0]=np.min(image[image!=0])
image_norm=(image-mean)/std

padded_image=ImagePadSym(image_norm)
big=Apply(ModelTo3D_single,model,data=padded_image[np.newaxis,...],input_size=target_size,padding=(20,20,20))

bigg=big*120
bigg=bigg.astype(np.uint8)[...,50:150]


bigg=np.swapaxes(bigg,0,-1)
bigg=np.swapaxes(bigg,1,2)
img=sitk.GetImageFromArray(bigg)
sitk.WriteImage(img, folder_name+'/big.mha')
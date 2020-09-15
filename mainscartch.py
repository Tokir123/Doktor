import numpy as np
from PIL import Image
from CnM.generators import *
from CnM.models import *
import SimpleITK as sitk
import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import keras.backend as K

#export TF_ENABLE_AUTO_MIXED_PRECISION=1

#K.set_floatx('float32')

print('d')
image = sitk.ReadImage('images/training/source/image.mha')
print('d')
image = np.moveaxis(sitk.GetArrayFromImage(image),0,-1)
#image=np.array(image)
labels=np.array(Image.open('images/training/source/labels.png'))
print(labels.shape)
dims=image.shape
print(dims)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

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

dataGenerator=dataGen(image, labels, lower=min, upper=max, target_size=48, batch_size=8)

myGen=transGenerator(dataGenerator)

model=unet3D(input_size=(48,48,48,1),outputSize=3,activation='softmax',loss='categorical_crossentropy')
model=addWeightTo3DModel(model, keras.losses.categorical_crossentropy)
l=model.fit_generator(myGen,steps_per_epoch=2000,epochs=1)
pred_cutouts_low=np.array((np.floor(0.75*dims[0]),500,20),dtype=int)
pred_cutouts_high=np.array((np.floor(0.75*dims[0])+47,547,67),dtype=int)
pred_cutouts_low=pred_cutouts_low[np.newaxis,...]
pred_cutouts_high=pred_cutouts_high[np.newaxis,...]
pred_image = getCutOut(image,pred_cutouts_low, pred_cutouts_high)

pred_labels = getCutOut(labels, pred_cutouts_low, pred_cutouts_high)
pred_weights = 1 * ((pred_labels[...,0] + pred_labels[...,1] + pred_labels[...,2]) == 1)
pred_weights=np.zeros(shape=pred_image.shape)+1
A={'input_data':pred_image, 'input_weight': pred_weights}
K=model.predict(A)

K=K[...,0:3]
sol=np.argmax(K,axis=-1)[0,...]
sol=(100*sol).astype(np.uint8)
img64 = sitk.GetImageFromArray(sol)#,isVector=True)
sitk.WriteImage(img64,'K.mha')
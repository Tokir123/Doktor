import numpy as np
from PIL import Image
from CnM.generators import *
from CnM.models import *
from CnM.runtime_methods import *

import SimpleITK as sitk
import keras
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import keras.backend.tensorflow_backend as K
from keras.models import load_model
from numba import cuda
import time

#export TF_ENABLE_AUTO_MIXED_PRECISION=1

#K.set_floatx('float32')

####Import
lr=(1e-3)
for i in range(5):

# for i in (0,):
    i=i+1
    lr=(1e-3)/7
    print(i)
    if i>1:
        del model
        del l
        K.clear_session()
    #     time.sleep(15 * 60)
    #     cuda.select_device(0)
    #     cuda.close()
    #     time.sleep(15 * 60)
    image  = sitk.ReadImage('images/training/source/rotated/image.mha')
    labels = np.array(Image.open('images/training/source/rotated/labelsd.png'))
    ####Import Done

    image = np.moveaxis(sitk.GetArrayFromImage(image), 0, -1)
    mean = np.mean(image)
    std = np.std(image)
    image = (image - mean) / std
    dims = image.shape
    labels = MakeLabel(full_image_size=image.shape)
    size = 64
    depth = 64
    class_weights = (1, 1, 10)

    folder_name = 'normmm'
    label_folder = 'adw'

    image = image[..., np.newaxis]
    min = (0, 0, 0)
    max = dims
    target_size = (size, size, depth)

    downscale_factor = 2**i
    batch_size = 4#*downscale_factor

    dataGenerator=dataGen(image, labels, lower=min, upper=max, target_size=target_size, batch_size=batch_size, slice_label=50,class_weights=class_weights)

    myGen=transGenerator(dataGenerator)

    model=unet3D(downscale_factor=downscale_factor,kernelSize=3,input_size=target_size+(1,),outputSize=1
                 ,activation='softmax',loss='binary_crossentropy')
    model=addWeightTo3DModel(model, keras.losses.categorical_crossentropy,lr=lr)

    model.load_weights(folder_name+'/my_model_weights'+str(i)+'.h5')
    l=model.fit_generator(myGen,steps_per_epoch=20000,epochs=1)


    model.save_weights(folder_name+'/my_model_weights'+str(i)+'.h5')
    padded_image = ImagePadSym(image[..., 0])

    image = sitk.ReadImage('images/training/source/wda/image.mha')

    ####Import

    ####Import Done

    image = np.moveaxis(sitk.GetArrayFromImage(image), 0, -1)
    for j in range(9):
        image[..., j] = image[..., 9]
    mean = np.mean(image[image != 0])
    std = np.std(image[image != 0])
    image_norm = image
    image_norm[image == 0] = np.min(image[image != 0])
    image_norm = (image - mean) / std

    padded_image = ImagePadSym(image_norm)
    big = Apply(ModelTo3D_single, model, data=padded_image[np.newaxis, ...], input_size=target_size,
                padding=(20, 20, 20))

    bigg = big * 120
    bigg = bigg.astype(np.uint8)[..., 50:150]

    bigg = np.swapaxes(bigg, 0, -1)
    bigg = np.swapaxes(bigg, 1, 2)
    # np.savetxt(folder_name +'/'+ str(i)+'big.txt', bigg)

    if i==1:
        bigstack=np.empty((4,)+bigg.shape)
    bigstack[i-1,...]=bigg
    img = sitk.GetImageFromArray(bigg)
    sitk.WriteImage(img, folder_name +'/'+ str(i)+'big.mha')


biggg=average_pictures(bigstack,required_ratio=0.6)
img = sitk.GetImageFromArray(biggg)
sitk.WriteImage(img, folder_name +'/'+ str(7)+'big.mha')
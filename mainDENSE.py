import numpy as np
from PIL import Image
from CnM.generators import *
from CnM.models import *
from CnM.runtime_methods import *
from CnM.post_methods import *
from CnM.denseunet3d import *
from CnM.hybridnet import *
import SimpleITK as sitk
import keras
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import keras.backend.tensorflow_backend as K
from keras.models import load_model
from numba import cuda
import time
from CnM.cust_callback import *
#export TF_ENABLE_AUTO_MIXED_PRECISION=1

#K.set_floatx('float32')

####Import
os.chdir("..") #Moves up one level
print(os.getcwd())

lr=(1e-3)
different_models=5

for i in range(different_models):

    i=i+1
    lr=(1e-3)/10
    print(i)
    if i>1:
        del model
        del l
        K.clear_session()
    #     time.sleep(15 * 60)
    #     cuda.select_device(0)
    #     cuda.close()
    #     time.sleep(15 * 60)
    image  = sitk.ReadImage('training_data/image/image.mha')

    ####Import Done

    image = np.moveaxis(sitk.GetArrayFromImage(image), 0, -1)
    mean = np.mean(image)
    std = np.std(image)
    image = (image - mean) / std
    dims = image.shape
    labels = MakeLabel(full_image_size=image.shape)
    weights = MakeWeight(full_image_size=image.shape)
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
    batch_size = 1#*downscale_factor

    dataGenerator=dataGen(image, labels, weights, lower=min, upper=max, target_size=target_size, batch_size=batch_size, slice_label=50,class_weights=class_weights)
    CallbackGenerator = dataGen(image, labels, weights, lower=min, upper=max, target_size=target_size, batch_size=2,
                                slice_label=50, class_weights=class_weights, callback_mode=True)
    myGen=transGenerator(dataGenerator)

    #model=denseunet_3d(input_size=(64, 64, 64, 1))
    model=dilated_resnet(input_size=(64, 64, 64, 1))

    model=addWeightTo3DModel(model, keras.losses.binary_crossentropy,lr=lr)
    #model.load_weights(folder_name + '/my_model_weights' + str(i) + '.h5')
    #model.load_weights(folder_name+'/model_best.hdf5')
    #model_checkpoint = tf.keras.callbacks.ModelCheckpoint('/weights.{epoch:02d}-{loss:.2f}.hdf5',
                                      # monitor='loss', verbose=1,
                                       #save_best_only=False, save_weights_only=False, mode='min', period=1)
    validation = ValidationCallback(CallbackGenerator, patience=10)
    l=model.fit_generator(myGen,steps_per_epoch=100,epochs=100,callbacks=[validation])


    model.save_weights(folder_name+'/my_model_weights'+str(i)+'.hdf5')
    padded_image = ImagePadSym(image[..., 0])

    #image = sitk.ReadImage('images/training/source/wda/image.mha')
    image  = sitk.ReadImage('training_data/image/image.mha')
    ####Import

    ####Import Done

    image = np.moveaxis(sitk.GetArrayFromImage(image), 0, -1)
    for j in range(9):
        image[..., j] = image[..., 9+(9-i)]

    mean = np.mean(image)
    std = np.std(image)
    image = (image - mean) / std
    image_norm=image
#    mean = np.mean(image[image != 0])
#    std = np.std(image[image != 0])
#    image_norm = image
#    image_norm[image == 0] = np.min(image[image != 0])
#    image_norm = (image - mean) / std

    padded_image = ImagePadSym(image_norm)
    big = Apply(ModelTo3D_single, model, data=padded_image[np.newaxis, ...], input_size=target_size,
                padding=(14, 14, 14))

    bigg = big * 120
    bigg = bigg.astype(np.uint8)[..., 50:150]

    bigg = np.swapaxes(bigg, 0, -1)
    bigg = np.swapaxes(bigg, 1, 2)
    # np.savetxt(folder_name +'/'+ str(i)+'big.txt', bigg)

    if i==1:
        bigstack=np.empty((different_models,)+bigg.shape)
    bigstack[i-1,...]=bigg
    img = sitk.GetImageFromArray(bigg)
    sitk.WriteImage(img, folder_name +'/'+ str(i)+'big.mha')


#biggg=average_pictures(bigstack,required_ratio=0.6)
biggg=np.sum(bigstack,axis=0)/5
biggg=biggg.astype(np.uint8)
img = sitk.GetImageFromArray(biggg)
sitk.WriteImage(img, folder_name +'/'+ str(7)+'big.mha')
# 400 355 18
#680 260 10
#900 360 35
x=520
y=180
z=10
new_picture=image[x:x+64,y:y+64,z:z+64]
sitk.WriteImage(sitk.GetImageFromArray(new_picture), 'validation/images/'+ str(2)+'.mha')
new_picture=labels[x:x+64,y:y+64,z:z+64]
sitk.WriteImage(sitk.GetImageFromArray(new_picture), 'validation/labels/'+ str(2)+'.mha')
new_picture=weights[x:x+64,y:y+64,z:z+64]
sitk.WriteImage(sitk.GetImageFromArray(new_picture), 'validation/weights/'+ str(2)+'.mha')

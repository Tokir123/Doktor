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
#from tensorflow.keras.losses import BinaryCrossentropy
from CnM.cust_callback import *
import copy
#export TF_ENABLE_AUTO_MIXED_PRECISION=1
tf.config.experimental.list_physical_devices('GPU')
#K.set_floatx('float32')

####Import
os.chdir("..") #Moves up one level
print(os.getcwd())
def training_step(*args)
    deformations=arg.deformations
    lr=arg.lr
    image_path=arg.image_path
    difficulty=arg.difficulty
    target_size=arg.target_size
    batch_size =arg.batch_size
    brightness_difficulty=arg.brightness_difficulty
    image  = sitk.ReadImage(image_path)

    ####Import Done

    image = np.moveaxis(sitk.GetArrayFromImage(image), 0, -1)
    mean = np.mean(image)
    std = np.std(image)
    image = (image - mean) / std
    dims = image.shape
    labels = MakeLabel(full_image_size=image.shape)
    weights = MakeWeight(full_image_size=image.shape, difficulty=difficulty)






    image = image[..., np.newaxis]
    min = (0, 0, 0)
    max = dims

    dataGenerator=dataGen(image, labels, weights, lower=min, upper=max, target_size=target_size, batch_size=batch_size, slice_label=50,class_weights=class_weights,callback_mode=False)

    myGen=transGenerator(dataGenerator,deformations=deformations)


    model=dilated_resnet_pad(input_size=(64, 64, 64, 1), mode=True)


    model=addWeightTo3DModel(model, tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE) ,lr=lr)
    model.load_weights( 'model_weights/6.565866364349651.h5')
    image_val=image[90:634,320:864,18:82,0]
    weights_val = copy.deepcopy(weights[90:634, 320:864, 18:82, 0])
    correcting = np.zeros(shape=weights_val.shape)
    correcting[15:weights_val.shape[0] - 15, 15:weights_val.shape[1] - 15, 15:weights_val.shape[2] - 15] = 1
    weights_val=weights_val*correcting

    labels_val = labels[90:634, 320:864, 18:82, 0]

    validation = ValidationCallback(CallbackGenerator,image_val,weights_val,labels_val, patience=15)
    l=model.fit_generator(myGen,steps_per_epoch=800,epochs=100,callbacks=[validation])


    model.save_weights(folder_name+'/my_model_weights'+str(i)+'.h5')

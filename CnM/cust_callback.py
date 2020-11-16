from CnM.generators import *
import keras
import tensorflow as tf
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
import copy
from CnM.generators import *
class ValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self,generator, image_val, weights_val,labels_val , patience=0,downscale_factor=1
                 ):
        super(ValidationCallback, self).__init__()
        self.downscale_factor = downscale_factor
        self.image_val=image_val
        self.weights_val=weights_val

        self.labels_val = labels_val

        self.patience = patience
        self.best = np.Inf
        self.best_weights = None
        self.max_resets=5
        self.resets=0
        self.im=[]
        self.la = []
        self.val_size=1

        for i in range(self.val_size):
            self.validation_X,self.validation_Y=next(generator)
            self.im.append(self.validation_X)
            self.la.append(self.validation_Y)
        self.generator=generator
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        #self.best = np.PZERO
        self.val_uas = []

    def on_epoch_end(self, epoch, logs=None):
        #val=0
        #with tf.device('/cpu:0'):
        #    for i in range(self.val_size):
        #        val=val+self.model.evaluate(self.im[i], self.la[i],verbose=0)[0]
        val=prediction_model(self.model, self.image_val, self.weights_val, self.labels_val,verbose=True,downscale_factor=self.downscale_factor)


        #val=val/self.val_size
        current=val
        ua_score = val
        self.val_uas.append(ua_score)
        logs['val_ua'] = ua_score

        if np.less(val, self.best):
            print(self.wait, self.patience)
            print(current, self.best)
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
            self.resets=0
            self.model.save_weights('model_weights/' + str(val) + '.h5')
        else:
            self.wait += 1
            print('Not good enough')
            print(self.wait, self.patience)
            print(current, self.best)
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model_stop_training = True
                print(self.wait, self.patience)
                print(current, self.best)
                print('Restoring model weights from the end of the best epoch. Amount of resets is now')
                print(self.resets)
                self.model.set_weights(self.best_weights)
                self.wait=0
                self.resets+=1
            '''
            if(self.resets>=self.max_resets):
                #self.val_size = self.val_size+50
                print("New validation data")
                self.im = []
                self.la = []
                for i in range(self.val_size):
                    print(i)
                    self.validation_X, self.validation_Y = next(self.generator)
                    self.im.append(self.validation_X)
                    self.la.append(self.validation_Y)
                self.resets=0
                print("Calculating new best score")
                #val = 0
                #with tf.device('/cpu:0'):
                #    for i in range(self.val_size):
                #        val = val + self.model.evaluate(self.im[i], self.la[i], verbose=0)[0]

                #val = val / self.val_size

                self.best=verification_model(self.model, self.image_val, self.weights_val, self.labels_val)
                print(self.best)
                '''


def verification_model(model,image,weights,labels):

    shape=copy.deepcopy(image.shape)
    image=image[np.newaxis,...,np.newaxis]
    labels = labels[np.newaxis, ..., np.newaxis]
    weights = weights[np.newaxis, ..., np.newaxis]
    A= {'input_data': image, 'input_weight': weights}
    with tf.device('/cpu:0'):
        big_model = dilated_resnet(input_size=shape + (1,))
        big_model = addWeightTo3DModel(big_model, tf.keras.losses.BinaryCrossentropy(), lr=1e-4)
        big_model.set_weights(model.get_weights())
        val=big_model.evaluate(A,labels)[0]
    return val

def prediction_model(model,image,weights,labels,verbose=False,downscale_factor=1):

    shape=copy.deepcopy(image.shape)
    image=image[np.newaxis,...,np.newaxis]
    labels = labels[np.newaxis, ..., np.newaxis]
    weights = weights[np.newaxis, ..., np.newaxis]
    A= {'input_data': image, 'input_weight': weights}
    with tf.device('/cpu:0'):
        #big_model = dilated_resnet_pad(input_size=shape + (1,),downscale_factor=downscale_factor)
        big_model = unet3D_pad(input_size=shape + (1,),downscale_factor=downscale_factor)
        big_model = addWeightTo3DModel(big_model, tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE), lr=1e-4)
        big_model.set_weights(model.get_weights())
        val=big_model.predict(A)
    import pdb
    #pdb.set_trace()
    ph=val[...,0]
    lo_f=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    lo=lo_f(labels, ph[...,np.newaxis])
    print("loss shape is")
    print(lo.shape)

    lo=lo[...,np.newaxis]
    score=lo*A.get('input_weight')
    score=np.sum(score)/np.sum(A.get('input_weight'))



    if(verbose):
        matrix=copy.deepcopy(val[0,...,0])

        matrix = matrix * 240/np.max(matrix)
        matrix =matrix.astype(np.uint8)

        matrix = np.swapaxes(matrix, 0, -1)
        matrix = np.swapaxes(matrix, 1, 2)

        img = sitk.GetImageFromArray(matrix)
        sitk.WriteImage(img, 'inference/predictions' + '/' + str(score) + '.mha')

        matrix = copy.deepcopy(lo.numpy())
        matrix=matrix[0,...,0]

        matrix = matrix * 240 / np.max(matrix)
        matrix = matrix.astype(np.uint8)

        matrix = np.swapaxes(matrix, 0, -1)
        matrix = np.swapaxes(matrix, 1, 2)

        img = sitk.GetImageFromArray(matrix)
        sitk.WriteImage(img, 'inference/losses' + '/' + str(score) + '.mha')
        matrix = copy.deepcopy(lo.numpy())
        matrix=matrix*A.get("input_weight")
        matrix = matrix[0, ..., 0]

        matrix = matrix * 240 / np.max(matrix)
        matrix = matrix.astype(np.uint8)

        matrix = np.swapaxes(matrix, 0, -1)
        matrix = np.swapaxes(matrix, 1, 2)

        img = sitk.GetImageFromArray(matrix)
        sitk.WriteImage(img, 'inference/weighted_losses' + '/' + str(score) + '.mha')
    return score


def big_prediction_model( image,weights_file,downscale_factor=2, target_size=(304,304,96), verbose=False):

    data = ImagePadSym(image)

    for j in range(2):
        transform = Identity3D()
        for i in range(5):
            transform = ConcetenateTrafo3D(
                ConcetenateTrafo3D(transform, Mirror3D(randint(0, 2))), Rotation3D(axis=randint(0, 2), k=randint(0, 3)))


        trans_dat=transform.transform(data)
        trans_pad=transform.transform(np.zeros((51,51,24)))
        trans_target_size = transform.transform(np.zeros(target_size))
        big_model = unet3D_pad(input_size=trans_target_size.shape, batch_size=1, downscale_factor=downscale_factor)
        big_model.load_weights(weights_file)
        with tf.device('/cpu:0'):
            big = Apply_model(big_model.predict, data=trans_dat[np.newaxis, ...,np.newaxis], input_size=trans_target_size.shape,
                              padding=trans_pad.shape)[0,...]
        if(j==0):
            res=transform.inverseTransform(big)
        else:
            res=res+transform.inverseTransform(big)


    res=res/2
    matrix = copy.deepcopy(res[ ...])

    matrix = matrix * 240 / np.max(matrix)
    matrix = matrix.astype(np.uint8)

    matrix = np.swapaxes(matrix, 0, -1)
    matrix = np.swapaxes(matrix, 1, 2)

    img = sitk.GetImageFromArray(matrix)
    sitk.WriteImage(img, 'inference/full' + '/' + str(np.random.random()) + '.mha')

    return

class CustomCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        val = Validation(self.X, self.Y, self.Batch_dev, self.output_shape, self.model)
        _, _, ua_score = val.validate(epoch)
        self.val_uas.append(ua_score)
        logs['val_ua'] = ua_score
        current = logs.get('val_ua')
        if np.less(self.best, current):
            print(self.wait, self.patience)
            print(current, self.best)
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model_stop_training = True
                print(self.wait, self.patience)
                print(current, self.best)
                print('Restoring model weights from the end of the best epoch.')
                self.model.set_weights(self.best_weights)


    def on_epoch_end(self,folder, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

        '''import images, mask and weights
            predict images
            calculate weighted loss
            
        '''

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))



    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))

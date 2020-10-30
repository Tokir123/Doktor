from CnM.generators import *
import keras
import tensorflow as tf
class ValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self,generator,  patience=0):
        super(ValidationCallback, self).__init__()
        self.patience = patience
        self.best = np.Inf
        self.best_weights = None
        self.max_resets=5
        self.resets=0
        self.im=[]
        self.la = []
        self.val_size=150
        for i in range(self.val_size):
            print(i)
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
        val=0
        #with tf.device('/cpu:0'):
        for i in range(self.val_size):
            val=val+self.model.evaluate(self.im[i], self.la[i],verbose=0)[0]



        val=val/self.val_size
        val=val/2
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
            if(self.resets>=self.max_resets):
                self.val_size = self.val_size+50
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
                val = 0

                for i in range(self.val_size):
                    val = val + self.model.evaluate(self.im[i], self.la[i], verbose=0)[0]

                val = val / self.val_size
                val = val / 2
                self.best=val
                print(self.best)


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

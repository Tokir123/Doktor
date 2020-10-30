import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv3D, MaxPooling3D, Dropout, UpSampling3D, Concatenate,Layer, BatchNormalization
from keras.optimizers import Adam
import numpy as np


def unet3D(pretrained_weights=None, input_size=(256, 256, 256, 1), kernelSize=3, outputSize=1, activation='sigmoid',lr=1e-3,
           loss='binary_crossentropy',downscale_factor=1):
    """
    Creates a 3D U-Net with batch normalization
    padding is same, we avoid the padding issues by giving the outer elements in the image array a weight of 0
    :param pretrained_weights:
    :param input_size:
    :param kernelSize:
    :param outputSize:
    :param activation:
    :param loss:
    :return:
    """
    d=downscale_factor
    inputs = Input(input_size, name='input_data')
    #batch1=BatchNormalization(axis=-1)(inputs) #really?
    conv1 = Conv3D(64//d, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    batch1=BatchNormalization(axis=-1)(conv1)
    conv1 = Conv3D(64//d, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(batch1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    batch2 = BatchNormalization(axis=-1)(pool1)
    conv2 = Conv3D(128//d, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(batch2)
    batch2 = BatchNormalization(axis=-1)(conv2)
    conv2 = Conv3D(128//d, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(batch2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    batch3 = BatchNormalization(axis=-1)(pool2)
    conv3 = Conv3D(256//d, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(batch3)
    batch3 = BatchNormalization(axis=-1)(conv3)
    conv3 = Conv3D(256//d, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(batch3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    batch4 = BatchNormalization(axis=-1)(pool3)
    conv4 = Conv3D(512//d, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(batch4)
    batch4 = BatchNormalization(axis=-1)(conv4)
    conv4 = Conv3D(512//d, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(batch4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)
    batch5 = BatchNormalization(axis=-1)(pool4)
    conv5 = Conv3D(1024//d, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(batch5)
    batch5 = BatchNormalization(axis=-1)(conv5)
    conv5 = Conv3D(1024//d, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(batch5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv3D(512//d, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(drop5))
    merge6 = Concatenate(axis=4)([drop4, up6])
    batch6 = BatchNormalization(axis=-1)(merge6)
    conv6 = Conv3D(512//d, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(batch6)
    batch6 = BatchNormalization(axis=-1)(conv6)
    conv6 = Conv3D(512//d, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(batch6)

    up7 = Conv3D(256//d, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv6))
    merge7 = Concatenate(axis=4)([conv3, up7])
    batch7 = BatchNormalization(axis=-1)(merge7)
    conv7 = Conv3D(256//d, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(batch7)
    batch7 = BatchNormalization(axis=-1)(conv7)
    conv7 = Conv3D(256//d, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(batch7)

    up8 = Conv3D(128//d, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv7))
    merge8 = Concatenate(axis=4)([conv2, up8])
    batch8 = BatchNormalization(axis=-1)(merge8)
    conv8 = Conv3D(128//d, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(batch8)
    batch8 = BatchNormalization(axis=-1)(conv8)
    conv8 = Conv3D(128//d, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(batch8)

    up9 = Conv3D(64//d, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv8))
    merge9 = Concatenate(axis=4)([conv1, up9])
    batch9 = BatchNormalization(axis=-1)(merge9)
    conv9 = Conv3D(64//d, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(batch9)
    batch9 = BatchNormalization(axis=-1)(conv9)
    conv9 = Conv3D(64//d, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(batch9)

    conv10 = Conv3D(outputSize, 1, activation=activation)(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    opt = tf.keras.optimizers.Adam(lr=lr)
    #opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])


    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def addWeightTo3DModel(model, loss,lr=1e-4):
    """
    Add additional input layer to model (for inputing custom weights)

    these weights are important to emphazise border regions inbetween particles
    :param model:
    :param loss:
    :return:
    """
    input = model.input
    output = model.output

    newInput = Input(batch_shape=(1,model.input_shape[1], model.input_shape[2], model.input_shape[3], model.input_shape[4]),
                     name="input_weight")
    numChannels = model.output_shape[-1]

    print("agha")
    print(input.shape)
    print(newInput.shape)
    print(output.shape)
    newOutput = Concatenate(axis=-1)([output, newInput])

    newModel = Model(inputs=(input, newInput), outputs=newOutput)
    opt = tf.keras.optimizers.Adam(lr=lr)
    #opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    newLoss = lambda y_true, y: newLossFun3D(y_true, y, loss, numChannels)

    newModel.compile(optimizer=opt, loss=newLoss, metrics=['accuracy'])
    return newModel


def newLossFun3D(y_true, y, loss, numChannels):
    """
    Weighted loss function. The weights are encoded in the last channel of y
    :param y_true:
    :param y:
    :param loss:
    :param numChannels:
    :return:
    """
    prediction, weights = tf.split(y, [numChannels, 1], axis=-1)
    val = tf.expand_dims(loss(y_true, prediction),axis=-1)*weights
    val=(tf.math.reduce_sum(val))/(tf.math.reduce_sum(weights))
    return val


class SymmetricPaddingConv3D(Layer):
    def __init__(self, padSize, filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,):
        super(SymmetricPaddingConv3D, self).__init__()
        self.padSize=padSize
        paddings=np.zeros(shape=(4,2),dtype="int32")
        paddings[1,0]=padSize
        paddings[1,1]=padSize
        paddings[2,0]=padSize
        paddings[2,1]=padSize
        paddings[3,0]=padSize
        paddings[3,1]=padSize
        self.paddings=tf.constant(paddings,dtype=tf.int32)

        self.Layer=Conv3D(filters,
                 kernel_size,
                 strides=strides,
                 padding=padding,
                 data_format=data_format,
                 dilation_rate=dilation_rate,
                 activation=activation,
                 use_bias=use_bias,
                 kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer,
                 kernel_regularizer=kernel_regularizer,
                 bias_regularizer=bias_regularizer,
                 activity_regularizer=activity_regularizer,
                 kernel_constraint=kernel_constraint,
                 bias_constraint=bias_constraint,)

        def call(self,input):
            in2=tf.pad(input,self.paddings,"SYMMETRIC")
            out=self.Layer(in2)
            out = out[:, self.padSize: self.padSize + input.shape[1],  self.padSize: self.padSize + input.shape[2],  self.padSize: self.padSize + input.shape[3],
                 :]
            return out

class SymmetricPadLayer(Layer):

    def __init__(self, padSize):
        self.padSize=padSize
        super(SymmetricPadLayer, self).__init__()

    def build(self, input_shape):
        paddings=np.zeros(shape=(5,2),dtype="int32")
        paddings[1,0]=self.padSize
        paddings[1,1]=self.padSize
        paddings[2,0]=self.padSize
        paddings[2,1]=self.padSize
        paddings[3,0]=self.padSize
        paddings[3,1]=self.padSize
        self.paddings=tf.constant(paddings,dtype=tf.int32)
        super(SymmetricPadLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.pad(inputs, self.paddings, "SYMMETRIC")

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.padSize*2+input_shape[1], self.padSize*2+input_shape[2], self.padSize*2+input_shape[3],input_shape[4])


class InverseSymmetricPadLayer(Layer):

    def __init__(self, padSize,insize,outChannelSize):
        self.padSize=padSize
        self.insize=insize
        self.outChannelSize=outChannelSize
        super(InverseSymmetricPadLayer, self).__init__()

    def build(self, input_shape):
        super(InverseSymmetricPadLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        out=inputs[:, self.padSize: self.padSize + self.insize[0], self.padSize: self.padSize + self.insize[1],
        self.padSize: self.padSize + self.insize[2],:]
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.insize[0],self.insize[1],self.insize[2],self.outChannelSize)


def SymmetricPaddingunet3D(pretrained_weights=None, input_size=(256, 256, 256, 1),padSize=8, kernelSize=3, outputSize=1, activation='sigmoid',lr=1e-3,
           loss='binary_crossentropy'):
    """
    Creates a 3D U-Net
    :param pretrained_weights:
    :param input_size:
    :param kernelSize:
    :param outputSize:
    :param activation:
    :param loss:
    :return:
    """





    inputs = Input(input_size, name='input_data')


    in2=SymmetricPadLayer(padSize)(inputs)
    conv1 = Conv3D(64, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(in2)
    conv1 = Conv3D(64, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = Conv3D(128, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv3D(128, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = Conv3D(256, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv3D(256, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = Conv3D(512, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv3D(512, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = Conv3D(1024, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv3D(1024, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv3D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(drop5))
    merge6 = Concatenate(axis=4)([drop4, up6])
    conv6 = Conv3D(512, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv3D(512, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv3D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv6))
    merge7 = Concatenate(axis=4)([conv3, up7])
    conv7 = Conv3D(256, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv3D(256, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv3D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv7))
    merge8 = Concatenate(axis=4)([conv2, up8])
    conv8 = Conv3D(128, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv3D(128, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv3D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling3D(size=(2, 2, 2))(conv8))
    merge9 = Concatenate(axis=4)([conv1, up9])
    conv9 = Conv3D(64, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv3D(64, kernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # conv9 = Conv2D(2, kernelSize, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = Conv3D(outputSize, 1, activation=activation)(conv9)

    out =InverseSymmetricPadLayer(padSize,input_size,outputSize)(conv10) #conv10[:, padSize: padSize + input_size[0], padSize: padSize +input_size[1],
        #  padSize: padSize + input_size[2],:]

    model = Model(inputs=inputs, outputs=out)

    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    opt = tf.keras.optimizers.Adam(lr=lr)
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    print("aada")
    print(conv10.shape)
    print(model.input.shape)
    print(model.output.shape)

    # model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

def UNetRoundDimension(num,depth=4):
    bas=2.0**4
    return int(np.ceil(num/bas)*bas)
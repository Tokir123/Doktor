from random import randint
from CnM.runtime_methods import *

import numpy as np


class BoxGenerator3D:

    def __init__(self, low, high, dataSize):
        """
        Creates a box generator which creates lower und upper coordinates of a box with lower coordinates low and
        upper coordinates high with size data
        :param low:
        :param high:
        :param dataSize:
        """

        self.low = np.array(low)
        self.high = (np.array(high) -np.array(dataSize))
        self.dataSize = np.array(dataSize)

    def includeSlice(self, sliceNumber=50):
        self.low[-1]= np.maximum(0,sliceNumber - self.dataSize[-1] + 1)
        self.high[-1]=np.minimum(self.high[-1], sliceNumber + self.dataSize[-1] -1)

        return(self)


    def getCoordinates(self, batch_size):
        """
        Generate batchsize boxes
        :param batchsize:
        :return:
        """

        corners = np.zeros(shape=(batch_size, 3), dtype="int")
        for i in range(batch_size):
            corners[i, :] = np.array([randint(self.low[0], self.high[0]), randint(self.low[1], self.high[1]),
                                      randint(self.low[2], self.high[2])])
            # identical to previous line but nicer looking
            # corners[i,:] =np.array([randint(self.low[j],self.high[j]) for j in range(3)])
        return corners, corners + self.dataSize - 1





class Transformation3D:
    def __init__(self):
        """
        Abstract class for bijective image coordinate system transformations
        """
        return

    def transform(self, image):
        """
        Transform coordinate system of image
        :param image:
        """
        pass

    def inverseTransform(self, image):
        """
        Inverse transform of coordinate system of image
        :param image:
        """
        pass


class Mirror3D(Transformation3D):

    def __init__(self, axis):
        """
        Transformation for mirroring image axis
        :param axis:
        """
        self.axis = axis

    def transform(self, image):
        out = np.flip(image, axis=self.axis)
        return out

    def inverseTransform(self, image):
        return self.transform(image)


class Rotation3D(Transformation3D):

    def __init__(self, axis, k):
        """
        Transformation for rotating image around axis by k*90 degrees
        :param axis:
        :param k:
        """
        self.axis = axis
        self.k = k

    def transform(self, image):
        return rot90(image, k=self.k, axis=self.axis)

    def inverseTransform(self, image):
        return rot90(image, k=4 - self.k, axis=self.axis)


class Identity3D(Transformation3D):
    def __init__(self):
        """
        Identity transformation
        """
        super().__init__()
        return

    def transform(self, image):
        return image

    def inverseTransform(self, image):
        return self.transform(image)


class ConcetenateTrafo3D(Transformation3D):

    def __init__(self, trafo1, trafo2):
        """
        Concetenate the image transformations trafo1 and trafo2
        :param trafo1:
        :param trafo2:
        """
        self.trafo1 = trafo1
        self.trafo2 = trafo2

    def transform(self, image):
        out = self.trafo1.transform(image)
        out = self.trafo2.transform(out)
        return out

    def inverseTransform(self, image):
        out = self.trafo2.inverseTransform(image)
        out = self.trafo1.inverseTransform(out)
        return out


def getConcetenateTrafo(trafos):
    """
    Concetenation of list of Transformation 3D objects
    :param trafos: list of Transformation3D objects
    :return: Transformation3D object which is the concettenated trafo of trafos
    """
    trafo = Identity3D()
    for i in range(len(trafos)):
        trafo = ConcetenateTrafo3D(trafo, trafos[i])
    return trafo


def mergeGenerators(gens):
    """
    Creates a generator from a list of generators gens. The new generator randomly picks a generator from the list
    and generates data from it
    :param gens:
    """
    while True:
        value = randint(0, len(gens) - 1)
        gen = gens[value]
        X, Y = gen.__next__()
        yield X, Y


def getCutOut(image, corner1, corner2):
    """
    Takes a cutout from image with lower and upper corners corner1 and corner 2
    :param image:
    :param corner1:
    :param corner2:
    :return:
    """
    tmp = corner2 - corner1 + 1
    tmp = tmp[0, :]
    cutouts = np.zeros(shape=(corner1.shape[0], tmp[0], tmp[1], tmp[2], image.shape[-1]))
    for i in range(corner1.shape[0]):
        c1 = corner1[i, :]
        c2 = corner2[i, :]
        cutouts[i, :, :, :, :] = image[c1[0]:c2[0] + 1, c1[1]:c2[1] + 1, c1[2]:c2[2] + 1, :]

    return cutouts

    # size=(corner2-corner1+1)[0,:]
    # slicing=[corner1[,]]

    # cutouts=np.array([image[corner1[0,i]:corner2[0,i] + 1, corner1[1,i]:corner2[1,i] + 1, corner1[2,i]:corner2[2,i] + 1,:] for i in range(corner1.shape[0])
    #    ])








def rot90(m, k=1, axis=2):
    """Rotate an array k*90 degrees in the counter-clockwise direction around the given axis"""
    m = np.swapaxes(m, 2, axis)
    m = np.rot90(m, k)
    m = np.swapaxes(m, 2, axis)
    return m


def weightedDataGenerator(batchsize, dataSize, data, weights, labels, low, high, sliceNumbers):
    """
    Creates a generator with two inputs. The random cutouts of size dataSize always contain a slice from sliceNumbers
    :param batchsize:
    :param dataSize:
    :param data:
    :param weights:
    :param labels:
    :param low:
    :param high:
    :param sliceNumbers:
    """
    gens = []

    for i in range(len(sliceNumbers)):
        gen = weightedDataGenerator_singleSlice(batchsize, dataSize, data, weights, labels, low, high, sliceNumbers[i])
        gens.append(gen)

    while True:
        value = randint(0, len(gens) - 1)
        X, Y = gens[value].__next__()
        yield X, Y


def weightedDataGenerator_singleSlice(batchsize, dataSize, data, weights, labels, low, high, sliceNumber):
    """
    Creates a generator with two inputs. The random cutouts of size dataSize always contain the sliceNumber-th slice
    :param batchsize:
    :param dataSize:
    :param data:
    :param weights:
    :param labels:
    :param low:
    :param high:
    :param sliceNumber:
    """
    boxgen = BoxGeneratorAroundSlice3D(low, high, dataSize, sliceNumber)
    while True:
        c1, c2 = boxgen.getCoordinates(batchsize)

        inp = getCutOut(data, c1, c2)
        weight = getCutOut(weights, c1, c2)

        label = getCutOut(labels, c1, c2)
        yield {'input_data': inp, 'input_weight': weight}, label


def WeightedMirrorAugmentor3D_All(gen):
    """
    Creates a generator which randomly mirrors an axis of the data
    :param gen:
    """
    gens = [gen, WeightedMirrorAugmentor3D(gen, axis=0), WeightedMirrorAugmentor3D(gen, axis=1),
            WeightedMirrorAugmentor3D(gen, axis=2)]
    gen = mergeGenerators(gens)
    while True:
        yield gen.__next__()


def WeightedMirrorAugmentor3D(gen, axis=0):
    """
    Creates a generator which mirrors the data at the axis axis+1
    :param gen:
    :param axis:
    """
    while True:
        X, Y = gen.__next__()
        X1 = X.get('input_data')
        X2 = X.get('input_weight')

        X1 = np.flip(X1, axis=axis + 1)
        X2 = np.flip(X2, axis=axis + 1)
        Y = np.flip(Y, axis=axis + 1)

        yield {'input_data': X1, 'input_weight': X2}, Y


def AnisotropicRotationAugmentorWeighted3D_All(gen):
    gens = [gen]
    for i in range(1):
        for j in range(1, 4):
            gens.append(RotationAugmentorWeighted3D(gen, j, axis=i))
    gen = mergeGenerators(gens)
    while True:
        yield gen.__next__()


def RotationAugmentorWeighted3D(gen, n, axis=0):
    """
    Creates a generator which rotates the data generated from gen by n*90 degrees around axis
    :param gen:
    :param n:
    :param axis:
    """
    while True:
        X, Y = gen.__next__()
        X1 = X.get('input_data')
        X2 = X.get('input_weight')

        X1 = rot90(X1, k=n, axis=axis)
        X2 = rot90(X2, k=n, axis=axis)

        Y = rot90(Y, k=n, axis=axis)

        yield {'input_data': X1, 'input_weight': X2}, Y


def transGenerator(gen):
    self.batch_size=gen.batch_size
    while True:
        # generate random vector
        # make random transformation
        # yield inverse(), transformed_input,transformed_label,label_layer
        transform = Identity3D()
        for i in range(5):
            transform = ConcetenateTrafo3D(
                ConcetenateTrafo3D(transform, Mirror3D(randint(0, 2))), Rotation3D(axis=randint(0, 2), k=randint(0, 3)))

        X, Y = gen.__next__()

        X1 = X.get('input_data')
        X2 = X.get('input_weight')
        X1 = np.array([transform.transform(X1[i, ...])
                       for i in range(X1.shape[0])
                       ])
        X2 = np.array([transform.transform(X2[i, ...])
                       for i in range(X2.shape[0])
                       ])
        Y = np.array([transform.transform(Y[i, ...])
                      for i in range(Y.shape[0])
                      ])

        yield {'input_data': X1, 'input_weight': X2}, Y

def transGenerator(gen):
    batch_size=gen.batch_size
    while True:
        # generate random vector
        # make random transformation
        # yield inverse(), transformed_input,transformed_label,label_layer
        #

        angle_list=np.random.uniform(size=(batch_size,3), high=360)


        X, Y = gen.__next__()

        X1 = X.get('input_data')
        X2 = X.get('input_weight')

        shape = X1[0, ..., 0].shape
        alpha = shape[1] * 10
        sigma = shape[1] * 0.08
        random_state = np.random.RandomState(None)

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dz = np.zeros_like(dx)
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        for i in range(batch_size):
            X1[i, ..., 0] = map_coordinates(X1[i, ..., 0], indices, order=2, mode='reflect').reshape(shape)
            X2[i, ..., 0] = map_coordinates(X2[i, ..., 0], indices, order=0, mode='reflect').reshape(shape)
            Y[i, ..., 0] = map_coordinates(Y[i, ..., 0], indices, order=0, mode='reflect').reshape(shape)


        X1 = rotation_3d(X1, angle_list)
        X2= rotation_3d(X2, angle_list)
        Y = rotation_3d(Y, angle_list)
        X2[...,0]=X2[...,0]*gen.ph




        #Image.fromarray(X1[0,:,:,3,0]).show
        yield {'input_data': X1, 'input_weight': X2}, Y


class dataGen(object):

    """"is a generator object that gives a generator that can work with model.fit_generator
    slice label is the the slice number for which we have a label, this slice will always be included in the generated box
    class weights is depreciated


    """
    def   __init__(self, image, labels, weights, lower, upper, target_size=(80,80,80), batch_size=1,slice_label=50, class_weights=(0.5,2,20),padding=25, callback_mode=False):
        self.batch_size = batch_size
        self.image = image
        self.labels = labels
        self.weights = weights
        self.lower = lower
        self.upper = upper
        self.target_size = target_size
        self.slice_label = slice_label
        self.class_weights = class_weights
        self.padding=padding
        self.ph=np.zeros(shape=(self.batch_size,)+self.target_size)
        self.ph[:,12:self.target_size[0]-12,12:self.target_size[0]-12,12:self.target_size[0]-12]=1
        self.callback_mode=callback_mode
    def __next__(self):
        cutouts_low, cutouts_high =BoxGenerator3D(low=self.lower, high=self.upper,
                                                   dataSize=self.target_size).includeSlice(self.slice_label).getCoordinates(batch_size=self.batch_size)
        rand = np.random.random(1)

        rand = 1 + (rand - 0.5)/5
        slice=randint(0,self.target_size[0]-1)
        image = getCutOut(self.image, cutouts_low, cutouts_high)
        weights=getCutOut(self.weights, cutouts_low, cutouts_high)
        mean = np.mean(image[0,slice,slice,...])
        std = np.std(image[0,slice,slice,...])
        if std==0:
            std=1
        
        image = (image - mean)
        image=image*rand
        labels = getCutOut(self.labels, cutouts_low, cutouts_high)
        if(self.callback_mode):
            weights[..., 0]=weights[..., 0]*self.ph


        ######
        #weights=weights*self.ph was moved to trasngen
        return {'input_data': image, 'input_weight': weights}, labels

def CorrectGen(gen,model):
    """depreciated

    """

    while True:
        # generate random vector
        # make random transformation
        # yield inverse(), transformed_input,transformed_label,label_layer
        #




        X, Y = gen.__next__()

        X1 = X.get('input_data')
        X2 = X.get('input_weight')
        X1 =  random_rotation_3d(X1, angle_list)
        X2 = random_rotation_3d(X2, angle_list)
        Y = random_rotation_3d(Y, angle_list)

        X1=model.predict(X)
        X1[X1==2]=0
        ph=np.zeros(shape=Y.shape)
        ph[X1==1 & Y==0]=1
        yield {'input_data': X1, 'input_weight': X2}, Y
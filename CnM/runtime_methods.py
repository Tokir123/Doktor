import numpy as np
import SimpleITK as sitk
import scipy
from numpy import shape
from PIL import Image
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from CnM.generators import *
import os
import copy
def OneHotEncoding(image):

    uq = np.unique(image)
    out = np.zeros(shape=(image.shape + (len(uq),)), dtype='float32')
    for i in range(len(uq)):
        out[image == uq[i], i] = 1
    return out
'''\
def   MakeLabel(label,full_image_size=(1008, 1204, 101), label_slices=50):

    onehot = OneHotEncoding(label)
    new_labels = np.zeros(full_image_size + (onehot.shape[-1],)) + 4
    new_labels[:, :, label_slices, :] = onehot
    return new_labels
'''
def   MakeLabel(label_folder='training_data/labels',full_image_size=(1008, 1204, 101)):
    '''
    :param label_folder:
    :param full_image_size:
    :return:
    '''
    file_list=os.listdir(label_folder)
    ph=np.zeros(shape=full_image_size+(1,))#+4
    for i in range(len(file_list)):
        file = file_list[i]
        label=np.array(Image.open(label_folder+'/'+file))
        if len(label.shape)==3:
            label=label[...,0]
        print(label.shape)
        print(np.mean(label))
        slice=int(file.split('.')[0])-1
        label=OneHotEncoding(label)[...,1]
        print(np.mean(label))
        ph[...,slice,:]=label[...,np.newaxis]

    print(ph.shape)
    print(np.unique(ph))
    print(np.mean(ph))
    return ph

def   MakeWeight(label_folder='training_data/weights',full_image_size=(1008, 1204, 101)):
    '''
    :param label_folder:
    :param full_image_size:
    :return:
    '''
    file_list=os.listdir(label_folder)
    ph=np.zeros(shape=full_image_size+(1,))
    for i in range(len(file_list)):
        file = file_list[i]
        weight=np.array(Image.open(label_folder+'/'+file))
        slice=int(file.split('.')[0])-1
        weight=weight/14

        weight=weight*weight+0.2

        ph[...,slice,0]=weight



    return ph
def HotToFile(input, file='placeholder.mha'):
    factor = 250/input.shape[-1]

    solution = np.argmax(input, axis=-1)
    solution = (factor * solution).astype(np.uint8)
    img = sitk.GetImageFromArray(solution)  # ,isVector=True)
    sitk.WriteImage(img, file)
    return

def ModelToFile(model,image,file_path='',file_name='placeholder'):
    pred_image=image[0,...,np.newaxis]
    pred_weights = np.zeros(shape=pred_image.shape) + 1
    A = {'input_data': pred_image, 'input_weight': pred_weights}
    output_OH=model.predict(A)
    exclude_weights=output_OH.shape[-1]-1
    for i in range(output_OH.shape[0]):
        HotToFile(input=output_OH[i,...,0:exclude_weights],file=file_path+file_name+str(i)+'.mha')

    return

def BatchToOutput(model,image):


    pred_image = image[...,np.newaxis]

    pred_weights = np.zeros(shape=pred_image.shape) + 1
    A = {'input_data': pred_image, 'input_weight': pred_weights}
    output_OH=model.predict(A)
    exclude_weights=output_OH.shape[-1]-1
    output=output_OH[...,0:exclude_weights]

    return output

def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """

    image_data=image[0,...,0]
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image_data.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    image_data=map_coordinates(image_data, indices, order=0, mode='reflect').reshape(shape)
    image[0,...,0]=image_data
    return map_coordinates(image, indices, order=0, mode='reflect').reshape(shape)

def ModelTo3D_single(model,image):

    pred_image = image[ ..., np.newaxis]
    pred_weights = np.zeros(shape=pred_image.shape) + 1
    A = {'input_data': pred_image, 'input_weight': pred_weights}
    output_OH=model.predict(A)
    exclude_weights=output_OH.shape[-1]-1
    output=output_OH[0,...,0]#:exclude_weights]
    #output=np.argmax(output,axis=-1)

    return output


def MakeFun(model,function):
    def f(image):
        return function(model,image)
    return f

#reset Keras Session
def reset_keras():
    sess = tf.compat.v1.keras.backend.get_session()
    tf.compat.v1.keras.backend.clear_session()
    sess.close()
    sess = tf.compat.v1.keras.backend.get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

def Apply(function,model, data, input_size, padding=(0, 0, 0)):
    step_size = np.array(input_size) - 2 * np.array(padding)
   #Assume data is of shape(batch_size,...)
    f = function
    dim = data.shape[1:4]
    output = np.zeros(dim)
    i = 0
    j = 0
    k = 0
    while True:
        if (i > (dim[0] - input_size[0])):
            i = 0
            j += step_size[1]
            print(j)
            if (j > (dim[1] - input_size[1])):
                j = 0
                k += step_size[2]
                if (k > (dim[2] - input_size[2])):
                    break  # brich ab

        else:
            if(np.mean(data[:,i:i + input_size[0], j:j + input_size[1], k:k + input_size[2]])!=0):
                output[i + padding[0]:i - padding[0] + input_size[0], j + padding[1]:j - padding[1] + input_size[1],
                    k + padding[2]:k - padding[2] + input_size[2]] =f(model,data[:,i:i + input_size[0], j:j + input_size[1], k:k + input_size[2]])[
                    padding[0]:input_size[0] - padding[0], padding[1]:input_size[1] - padding[1],
                    padding[2]:input_size[2] - padding[2]]

        i += step_size[0]
    return output




def TTA3D(image, model, transformations):
    """
    Compute Test Time Augmentation of image using transformations
    :param image:
    :param model:
    :param transformations:
    :return: model's output for len(transformations) transformations of image

    takes a resxresxres picture
    """
    # for j in range(3)])
    # out=np.moveaxis(out,0,-1)
    # inp=np.zeros(shape=(len(transformations),)+image.shape)

    inp= [transformations[i].transform(image) for i in range(len(transformations))]

    inp = np.array( inp)




    out = BatchToOutput(model,inp)

    out = np.array([np.array(transformations[i].inverseTransform(out[i, ...])) for i in range(len(transformations))
                    ])
    return out


def TTAHelper(model, image):
    image=image[0,...]

    transform_list=[]
    for j in range(4):
        transform = Identity3D()
        for i in range(5):
            transform = ConcetenateTrafo3D(
                ConcetenateTrafo3D(transform, Mirror3D(randint(0, 2))), Rotation3D(axis=randint(0, 2), k=randint(0, 3)))
        transform_list.append(transform)

    out=TTA3D(image,model,transform_list)
    out1=np.mean(out[...,0:2],axis=0)
    out2=np.max(out[...,2],axis=0)
    out2=out2[...,np.newaxis]

    out=np.concatenate([out1, out2],axis=-1)
    out=np.argmax(out,axis=-1)
    return out

def rotation_3d(batch, angle_list):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).

    Arguments:
    max_angle: `float`. The maximum rotation angle.

    Returns:
    batch of rotated 3D images
    """
    if len(batch.shape)==3:
        batch=np.array([batch[...,np.newaxis] for i in range(angle_list.shape[0])])


    batch_rot = np.zeros(batch.shape)
    for i in range(batch.shape[0]):
        for j in range(batch.shape[-1]):
            image1 = batch[i,...,j]
            angle = angle_list[i,0]
            image1 = scipy.ndimage.rotate(image1, angle,order=0, mode='reflect', axes=(0, 1), reshape=False,prefilter=False)

            # rotate along y-axis
            angle = angle_list[i,1]
            image1 = scipy.ndimage.rotate(image1, angle,order=0, mode='reflect', axes=(0, 2), reshape=False, prefilter=False
                                          )

            # rotate along x-axis
            angle = angle_list[i,2]
            batch_rot[i,...,j] = scipy.ndimage.rotate(image1, angle,order=0, mode='reflect', axes=(1, 2), reshape=False,prefilter=False)
            #                print(i)

    return batch_rot
def inverse_rotation_3d(batch, angle_list):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).

    Arguments:
    max_angle: `float`. The maximum rotation angle.

    Returns:
    batch of rotated 3D images
    """


    batch_rot = np.zeros(batch.shape)
    for i in range(batch.shape[0]):
        for j in range(batch.shape[-1]):
            image1 = batch[i,...,j]
            angle = 360-angle_list[i, 2]
            image1= scipy.ndimage.rotate(image1, angle, order=0, mode='nearest', axes=(1, 2),
                                                        reshape=False, prefilter=False)
            angle = 360-angle_list[i, 1]
            image1 = scipy.ndimage.rotate(image1, angle, order=0, mode='nearest', axes=(0, 2), reshape=False,
                                          prefilter=False)
            angle = 360-angle_list[i,0]
            batch_rot[i, ..., j] = scipy.ndimage.rotate(image1, angle,order=0, mode='nearest', axes=(0, 1), reshape=False,prefilter=False)

    return batch_rot


"""
class dataGen(object):
    def __init__(self, model_list):
        pass
        self.model_list = model.list
    def predict(self, image_batch):
        pass
        output=np.zeros(shape=image.shape+(1,))
        for i in range(len(self.model_list)):

def import_image_labels(image_filename,label_filename):
    image = sitk.ReadImage(image_filename)
    labels=np.array(Image.open(label_filename))
    ####Import Done

    image = np.moveaxis(sitk.GetArrayFromImage(image),0,-1)

    image=image/(np.max(image))
    image[image==0]=np.mean(image)
    dims=image.shape
    labels=MakeLabel(labels)

    image=image[...,np.newaxis]
    return image,labels

"""
def another_TTA(tta_batches,target_size,image,model,name):

    randoms=np.random.uniform(size=(tta_batches,3), high=360)
    image=rotation_3d(image[...,0],randoms)

    ph=np.array([Apply(ModelTo3D_single,model,data=image[[i],...,0],input_size=target_size,padding=(8,8,8)) for i in range(tta_batches)]).astype(int)
    del image
    ph=inverse_rotation_3d(ph[...,np.newaxis],randoms)

    for i in range(tta_batches):
        bigg = ph[i,...,0] * 120
        bigg = bigg.astype(np.uint8)

        bigg = np.swapaxes(bigg, 0, -1)
        bigg = np.swapaxes(bigg, 1, 2)

        bigg= sitk.GetImageFromArray(bigg)
        sitk.WriteImage(bigg, name)
    return




def ImagePadSym(image):
    ''':arg
     image.shape[2] has to  be divisible by 2
    '''

    out=np.zeros(shape=image.shape[0:2]+(image.shape[2]*2-2,))
    out[...,int(out.shape[2]*0.25):int(out.shape[2]*0.75+1)]=image
    out[...,int(out.shape[2]*0.75+1):]=np.flip(image,axis=2)[...,:int((image.shape[2]-1)*0.5-1)]
    out[..., :int(out.shape[2] * 0.25)] = np.flip(image,axis=2)[..., int(image.shape[2] * 0.5 +1):]

    return out

def average_pictures(image_stack,required_ratio=0.5):
    ''':arg

    2 means white, 0 means black, 1 means grey for input
    required_ratio is the ratio of required pictures to be grey for the aggregated pixel to be grey
    image_stack is a np.array of shape (stack_length,x,y,z)
    output is a np.array of shape (x,y,z) with entries in int8
    '''
    stack_shape=image_stack.shape
    stack_size=stack_shape[0]
    image_shape=stack_shape[1:]
    values=np.unique(image_stack)

    if(len(values)>3):
        print("more than 3 values")
    grey=values[1]


    image_stack=image_stack==grey
    output=np.zeros(image_shape)
    ''':returns
    an output image
    '''
    for i in range(stack_size):
       output=output+image_stack[i,...]

    output=output>(stack_size*required_ratio)
    output=output*120
    return output.astype(np.uint8)


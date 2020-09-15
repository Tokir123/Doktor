from multiprocessing import Process
import time
import numpy as np
import SimpleITK as sitk
import scipy
from numpy import shape
from PIL import Image
from CnM.generators import *
from CnM.runtime_methods import *
import os
import multiprocessing as mp

def another_TTA(tta_batches,target_size,image,model,folder_name,pool):
    os.mkdir(folder_name)
    randoms=np.random.uniform(size=(tta_batches,3), high=360)
    image=rotation_3d(image[...,0],randoms)


    ph = np.array(
        [pool.apply(Apply, args=(ModelTo3D_single, model, image[[i], ..., 0], target_size, (8, 8, 8))) for i in
        range(tta_batches)])
    pool.close()


    ph=inverse_rotation_3d(ph[...,np.newaxis],randoms)

    for i in range(tta_batches):
        bigg = ph[i,...,0] * 120
        bigg = bigg.astype(np.uint8)

        bigg = np.swapaxes(bigg, 0, -1)
        bigg = np.swapaxes(bigg, 1, 2)
        img = sitk.GetImageFromArray(bigg)
        sitk.WriteImage(img, str(i)+'.mha')
    return
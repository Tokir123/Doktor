#Import Bild als image
#
#
#

def labelling_neighbours(classification,pixel):

    return class_indication



class_iterator=1
classification=np.zeros(image.shape)

for pixel in image:
    if (image[pixel]==1):

        if(labelling_neighbours==0):
            classification[pixel]=class_iterator
            class_iterator=class_iterator+1
        else:
            classification[pixel] = labelling_neighbours





weighting_all=np.zeros(image.shape+(length(np.unique(classification))-1))

for i in range(np.unique(classification))-1):
    weighting_all[...,i]=(classification==i)

for iteration in range[30]:
    for i in range(np.unique(classification))-1):
        if (weighting_all[pixel,i]==0):
            if(max_neighbor(weighting_all[, i], pixel)>0):
                weighting_all[pixel, i] = max_neighbor(weighting_all[, i], pixel)+1

weighting_all[weighting_all==0]=40
weighting_all=weighting_all-1

weighting=np.zeros(image.shape)

for pixel in image:
    if(image[pixel==0]):\
        weighting[pixel]=weight_function(np.unique(weighting_all[pixel,])[0],np.unique(weighting_all[pixel,])[1])
weighting=weighting+1

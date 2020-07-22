
import numpy as np
import os
import cv2


def add_noise(image, noise_typ):
    """
        Adds slight noise to the image
    """
    if noise_typ == "gauss":
       row,col,ch= image.shape
       mean = 0
       var = 0.1 * 255
       sigma = var**0.5
       gauss = np.random.normal(mean,sigma,(row,col,ch))
       gauss = gauss.reshape(row,col,ch)
       noisy = image + gauss
       return noisy
    elif noise_typ == "s&p":
       row,col,ch = image.shape
       s_vs_p = 0.5
       amount = 0.004
       out = np.copy(image)
       # Salt mode
       num_salt = np.ceil(amount * image.size * s_vs_p)
       coords = [np.random.randint(0, i - 1, int(num_salt))
               for i in image.shape]
       out[coords] = 1

       # Pepper mode
       num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
       coords = [np.random.randint(0, i - 1, int(num_pepper))
               for i in image.shape]
       out[coords] = 0
       return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy


def generate_shadow_coordinates(imshape, no_of_shadows=1):
        vertices_list=[]
        for index in range(no_of_shadows):
            vertex=[]
            for dimensions in range(np.random.randint(3,15)): ## Dimensionality of the shadow polygon
                vertex.append(( imshape[1]*np.random.uniform(),imshape[0]*np.random.uniform()))
            vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices
            vertices_list.append(vertices)
        return vertices_list ## List of shadow vertices

def add_shadow(image,no_of_shadows=1):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    mask = np.zeros_like(image)
    imshape = image.shape
    vertices_list= generate_shadow_coordinates(imshape, no_of_shadows) #3 getting list of shadow vertices
    for vertices in vertices_list:
        cv2.fillPoly(mask, vertices, 255) ## adding all shadow polygons on empty mask, single 255 denotes only red channel
    image_HLS[:,:,1][mask[:,:,0]==255] = image_HLS[:,:,1][mask[:,:,0]==255]*0.5   ## if red channel is hot, image's "Lightness" channel's brightness is lowered
    image_RGB = cv2.cvtColor(np.clip(image_HLS, 0, 255),cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return image_RGB

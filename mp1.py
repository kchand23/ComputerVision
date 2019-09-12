#Krishna Vamsi Chandu
#kchand23

import numpy as np
import cv2

#Opening a picture with OpenCV
art = cv2.imread('art.png',-1)
lena = cv2.imread('lena.png',-1)


# Helper function for convolution taking each layer one by one and return the convoluted layer.
def convolution_help(layer_in, kernel):
    k = int(kernel.shape[0] / 2)

    # create a padded image with zeroes in the border according to the size of the kernel.
    layer_out = np.zeros_like(layer_in)
    padded_image = np.zeros((layer_in.shape[0] + (k * 2), layer_in.shape[1] + (k * 2)))
    padded_image[k:-k, k:-k] = layer_in

    # Flip the kernel horizontally (lr) and vertically(ud) for convolution
    kernel = np.flipud(np.fliplr(kernel))

    # Iterate through every pixel
    for i in range(0, layer_in.shape[0]):
        for j in range(0, layer_in.shape[1]):
            layer_out[i, j] = (kernel * padded_image[i:i + kernel.shape[0], j:j + kernel.shape[0]]).sum()

    return layer_out

def convolution(img_in, kernel):
    #Grayscale images
    if(img_in.ndim == 2):
        return convolution_help(img_in, kernel)
    #RGB Imgaes
    else:
        img_out = np.zeros_like(img_in)
        for i in range(0, img_in.shape[2]):
            img_out[:,:,i] = convolution_help(img_in[:,:,i],kernel)
        return img_out


# Helper function for cross-correlation taking each layer one by one and return the correlated layer. Very similar to convolution without flipping the kernel
def correlation_help(layer_in, kernel):
    k = int(len(kernel) / 2)

    # create a padded image with zeroes in the border according to the size of the kernel.
    layer_out = np.zeros_like(layer_in)
    padded_image = np.zeros((layer_in.shape[0] + (k * 2), layer_in.shape[1] + (k * 2)))
    padded_image[:layer_in.shape[0], :layer_in.shape[1]] = layer_in

    # Iterate through every pixel
    for i in range(0, layer_in.shape[0]):
        for j in range(0, layer_in.shape[1]):
            layer_out[i, j] = (kernel * padded_image[i:i + kernel.shape[0], j:j + kernel.shape[0]]).sum()

    return layer_out

def correlation(img_in, kernel):
    #Grayscale images
    if(img_in.ndim == 2):
        return correlation_help(img_in, kernel)
    #RGB Imgaes
    else:
        img_out = np.zeros_like(img_in)
        for i in range(0, img_in.shape[2]):
            img_out[:,:,i] = correlation_help(img_in[:,:,i],kernel)
        return img_out


def median_filter(img_in, kernel_size):
    k = kernel_size

    # create a padded image with zeroes in the border according to the size of the kernel.
    img_out = np.zeros_like(img_in)
    padded_image = np.zeros((img_in.shape[0] + (k * 2), img_in.shape[1] + (k * 2)))
    padded_image[:img_in.shape[0], :img_in.shape[1]] = img_in

    # Iterate through every pixel
    for i in range(0, img_in.shape[0]):
        for j in range(0, img_in.shape[1]):
            # a window of size of the kernel.
            window = padded_image[i:i + kernel_size, j:j + kernel_size]
            # flatten the window into a single array to find the median.
            img_out[i, j] = (np.median(window.flatten()))
    return img_out

#Display an image

mean_filter_1 = np.array([[1 / 9, 1 / 9, 1 / 9],
               [1 / 9, 1 / 9, 1 / 9],
               [1 / 9, 1 / 9, 1 / 9]])

mean_filter_2 = np.array([[1/25, 1/25, 1/25,1/25,1/25],
               [1/25, 1/25, 1/25,1/25,1/25]
               ,[1/25, 1/25, 1/25,1/25,1/25],[1/25, 1/25, 1/25,1/25,1/25],[1/25, 1/25, 1/25,1/25,1/25]])

sharpening_filter_1 = np.array([[0, 0, 0],
               [0, 2 , 0],
               [0, 0, 0]]) - mean_filter_1


gaussian_filter_1 = np.array((
[0.077847,	0.123317,	0.077847],
[0.123317,	0.195346,	0.123317],
[0.077847,	0.123317,	0.077847]))

gaussian_filter_2 = np.array(([0.102059,	0.115349,	0.102059],
[0.115349,	0.130371,	0.115349],
[0.102059,	0.115349,	0.102059]))

gaussian_filter_3 = np.array(([0.003765,	0.015019,	0.023792,	0.015019,	0.003765],
[0.015019,	0.059912,	0.094907,	0.059912,	0.015019],
[0.023792,	0.094907,	0.150342,	0.094907,	0.023792],
[0.015019,	0.059912,	0.094907,	0.059912,	0.015019],
[0.003765,	0.015019,	0.023792,	0.015019,	0.003765]))

gaussian_filter_4 = np.array((
[0.023528,	0.033969,	0.038393,	0.033969,	0.023528],
[0.033969	,0.049045,	0.055432,	0.049045,	0.033969],
[0.038393,	0.055432,	0.062651,	0.055432,	0.038393],
[0.033969,	0.049045,	0.055432,	0.049045,	0.033969],
[0.023528,	0.033969,	0.038393,	0.033969,	0.023528]))
#cv2.imshow('image',correlation(art,mean_filter_2))
cv2.imshow('image',median_filter(art,11))
cv2.waitKey(0)
cv2.destroyAllWindows()
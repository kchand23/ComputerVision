# Krishna Vamsi Chandu
# kchand23


# Canny Edge Detector

import numpy as np
import cv2
import math


def conv_transform(image):
    image_copy = image.copy()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_copy[i][j] = image[image.shape[0] - i - 1][image.shape[1] - j - 1]

    return image_copy


def conv(image, kernel):
    kernel = conv_transform(kernel)
    image_h = image.shape[0]
    image_w = image.shape[1]

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    h = kernel_h // 2
    w = kernel_w // 2

    image_conv = np.zeros_like(image)

    for i in range(h, image_h - h):
        for j in range(w, image_w - w):
            sum = 0
            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum = sum + kernel[m][n] * image[i - h + m][j - w + n]

            image_conv[i][j] = int(sum)

    return image_conv


# Helper function for convolution taking each layer one by one and return the convoluted layer.
def convolution_help(layer_in, kernel):
    k = int(kernel.shape[0] / 2)

    # create a padded image with zeroes in the border according to the size of the kernel.
    layer_out = np.zeros_like(layer_in)
    padded_image_temp = np.zeros((layer_in.shape[0] + (k * 2), layer_in.shape[1] + (k * 2)))
    padded_image = np.zeros_like(padded_image_temp)
    padded_image[k:-k, k:-k] = layer_in

    # Flip the kernel horizontally (lr) and vertically(ud) for convolution
    kernel = np.flipud(np.fliplr(kernel))

    # Iterate through every pixel
    for i in range(0, layer_in.shape[0]):
        for j in range(0, layer_in.shape[1]):
            val = (kernel * padded_image[i:i + kernel.shape[0], j:j + kernel.shape[0]]).sum()
            layer_out[i, j] = val

    return layer_out


def convolution(img_in, kernel):
    # Grayscale images
    if (img_in.ndim == 2):
        return convolution_help(img_in, kernel)
    # RGB Imgaes
    else:
        img_out = np.zeros_like(img_in)

        for i in range(0, img_in.shape[2]):
            img_out[:, :, i] = convolution_help(img_in[:, :, i], kernel)
        return img_out


# Create a Gaussian Filter and apply convolution to the image using the same filter.
def gaussianSmoothing(img_in, kernel_size, sigma):
    k = int(kernel_size / 2)
    l = int(kernel_size / 2)

    temp = 0
    gaussian_kernel_1 = np.zeros((kernel_size, kernel_size))
    gaussian_kernel = np.zeros_like(gaussian_kernel_1)

    for i in range(-k, k + 1):
        for j in range(-l, l + 1):
            gaussian_kernel[i + k][j + l] = (1 / (2 * math.pi * (sigma ** 2))) * math.exp(
                -1 * (i ** 2 + j ** 2) / (2 * sigma ** 2))
            temp += gaussian_kernel[i + k][j + l]

    # Normalize the kernel
    for i in range(-k, k + 1):
        for j in range(-l, l + 1):
            gaussian_kernel[i + k][j + l] /= temp

    return convolution(img_in, gaussian_kernel)


def imageGradient(img_in):
    S_x = np.array((
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]))

    S_y = np.array((
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]))

    gradient_X = convolution(img_in, S_x)
    gradient_Y = convolution(img_in, S_y)



    mag_image = np.zeros_like(gradient_X)

    for i in range(0, gradient_X.shape[0]):
        for j in range(0, gradient_X.shape[1]):
            val = math.sqrt(gradient_X[i][j] ** 2 + gradient_Y[i][j] ** 2)
            mag_image[i][j] = val / 255.0

    tan_image = np.arctan2(gradient_Y, gradient_X)

    return mag_image, tan_image


def nonMaximaSuppress(mag, theta):
    supressed_image = np.zeros_like(mag)
    for i in range(0, mag.shape[0] - 1):
        for j in range(0, mag.shape[1] - 1):
            if mag[i][j] != 0:
                angle = theta[i][j]
                if angle < 0:
                    angle += math.pi
                if (0 <= angle < math.pi / 8) or ((7 * math.pi) / 8 <= angle <= math.pi):
                    pixel_1 = mag[i, j + 1]
                    pixel_2 = mag[i, j - 1]
                elif math.pi / 8 <= angle < (3 * math.pi) / 8:
                    pixel_1 = mag[i + 1, j - 1]
                    pixel_2 = mag[i - 1, j + 1]

                elif (3 * math.pi) / 8 <= angle < ((5 * math.pi) / 8):
                    pixel_1 = mag[i + 1, j]
                    pixel_2 = mag[i - 1, j]

                elif (5 * math.pi) / 8 <= angle < ((7 * math.pi) / 8):
                    pixel_1 = mag[i - 1, j - 1]
                    pixel_2 = mag[i + 1, j + 1]

                if mag[i][j] >= pixel_1 and mag[i][j] >= pixel_2:
                    supressed_image[i][j] = mag[i][j]
                else:
                    supressed_image[i][j] = 0
    return supressed_image


def threshold(image, threshold):
    low_threshold = threshold / 2
    mag_weak = np.zeros_like(image)
    mag_strong = np.zeros_like(image)

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i][j] > threshold:
                mag_strong[i][j] = image[i][j]
            elif image[i][j] < low_threshold:
                mag_strong[i][j] = 0
            elif low_threshold < image[i][j] < threshold:
                mag_weak[i][j] = image[i][j]
    return edgeLinking(mag_weak, mag_strong)


def edgeLinking(mag_weak, mag_strong):
    # img_out = np.zeros_like(mag_strong)
    for i in range(0, mag_weak.shape[0] - 1):
        for j in range(0, mag_weak.shape[1] - 1):
            if mag_weak[i][j] != 0:
                if ((mag_strong[i + 1, j - 1] != 0) or (mag_strong[i + 1, j] != 0) or (mag_strong[i + 1, j + 1] != 0)
                        or (mag_strong[i, j - 1] != 0) or (mag_strong[i, j + 1] != 0)
                        or (mag_strong[i - 1, j - 1] != 0) or (mag_strong[i - 1, j] != 0) or (
                                mag_strong[i - 1, j + 1] != 0)):
                    mag_strong[i][j] = mag_weak[i][j]
                    mag_weak[i][j] = 0
                else:
                    mag_strong[i][j] = 0
                    mag_weak[i][j] = 0
    for i in range(0, mag_weak.shape[0] - 1):
        for j in range(0, mag_weak.shape[1] - 1):
            print(mag_strong[i][j])

    return mag_strong


# Opening a picture with OpenCV
lena = (cv2.imread('lena_gray.png', 0)).astype(float)
test = (cv2.imread('test.png', 0)).astype(float)

smooth_image = gaussianSmoothing(lena, 3, 2)
mag, theta = imageGradient(smooth_image)
supressed = nonMaximaSuppress(mag, theta)
thresholded = threshold(supressed, 0.5)
cv2.imshow('Hysteris ', thresholded)
cv2.waitKey(0)
cv2.destroyAllWindows()


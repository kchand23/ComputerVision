import cv2
import numpy as np
from sklearn.preprocessing import normalize


def create_histogram():
    histogram = np.zeros([256, 256], dtype=np.float64)
    return histogram


def modify_histogram(hsv_image, histogram):
    for i in range(hsv_image.shape[0]):
        for j in range(hsv_image.shape[1]):
            # print(hsv_image[i][j])
            if (hsv_image[i][j][0] != 0 and hsv_image[i][j][1] != 0 and hsv_image[i][j][2] != 255):
                histogram[hsv_image[i][j][0]][hsv_image[i][j][1]] += 1
    return histogram


def normalize_histogram(histogram):
    histogram = (histogram) / histogram.max()
    return histogram


def detect(hsv_image, threshold, histogram):
    output = np.zeros_like(hsv_image)
    print(output.shape)
    for i in range(hsv_image.shape[0]):
        for j in range(hsv_image.shape[1]):
            if histogram[hsv_image[i][j][0]][hsv_image[i][j][1]] > threshold:
                output[i][j] = hsv_image[i][j]
            else:
                output[i][j] = np.array([0, 0, 0])
    output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    cv2.imshow('title', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Create empty histogram
histogram = create_histogram()

# Read training data
for i in range(1, 8):
    print(i)
    image = cv2.imread('Hand' + str(i) + '.jpg')
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Update the histogram matrix as we go through each pixel
    histogram = modify_histogram(hsv_image, histogram)

# Normalize Histogram
histogram = normalize_histogram(histogram)

# Read in the input image as HSV
test_image = cv2.imread('pointer1.bmp')
hsv_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)

# Use the histogram array to detect skin color tone of the given image.
detect(hsv_test_image, 0.003, histogram)

import cv2
import numpy as np
import glob

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from random import seed
from random import randint

import pickle

from collections import Counter


def kNearestNeighbors(k, histograms, descriptor):
    result = []

    for index, data in enumerate(histograms):
        dist = np.linalg.norm(data[0] - descriptor)
        result.append((data[1], dist))

    result.sort(key=lambda tup: tup[1])

    test_list = []
    for i in range(0, k):
        test_list.append(result[i][0][1])

    test_list = Counter(test_list)
    res = test_list.most_common(1)[0][0]
    return res


num_clusters = 300
prob = 3
kmeans = KMeans(n_clusters = num_clusters)

images = []

seed = 1

answers = []
count = 0

for i in glob.glob("data\\train\\*"):
    for img in glob.glob(i+"/*.jpg"):
        value = randint(0, 10)
        if(value < prob):
            n = cv2.imread(img)
            images.append(n)
            temp = img.split("\\")
            answers.append((count, temp[-2]))
            count += 1

sift = cv2.xfeatures2d.SIFT_create()

descriptors = []
keyDesc = []
keyPoints = []
for image in images:
    keypoint, descriptor = sift.detectAndCompute(image,None)

    if descriptor is not None:
        descriptors.append(descriptor)
        keyPoints.append(keypoint)
        keyDesc.append((keypoint, descriptor))
        #print(descriptor.shape)


vStack = np.array(descriptors[0])
for remaining in descriptors[1:]:
    vStack = np.vstack((vStack, remaining))

# Code for calculating kmeans, when file not present
# temp = kmeans.fit_predict(vStack)
# file = open('important' + str(num_clusters) + "_" + str(prob), 'wb')
# pickle.dump(kmeans, file)
# file.close()

kmeans = pickle.load( open( "important300_3", "rb" ) )

#Code for Visual Dictionary.

# match_count = 0
#
# for index, data in enumerate(temp):
#     if data == 30:
#         descriptor = vStack[index]
#         for i in keyDesc:
#             for num,j in enumerate(i[1]):
#                 if np.array_equal(j, descriptor):
#                     point = i[0][num]
#
#         point_x = int(point.pt[0])
#         point_y = int(point.pt[1])
#
#         patch_image = images[0][point_x:point_x + 100, point_y:point_y + 100]
#
#         cv2.imwrite('cluster_center_30' + str(match_count) + '.jpg', patch_image)
#         match_count += 1
#
# min = 1000000
# min_index_i = 0
# min_index_j = 0
# point = kmeans.cluster_centers_[1]
# print(point.shape)
# index_i = 0
# index_j = 0
# for i in keyDesc:
#     index_j = 0
#     for j in i[1]:
#         dist = np.linalg.norm(point - j)
#         #print(dist)
#         if dist < min:
#             min = dist
#             min_index_i = index_i
#             min_index_j = index_j
#         index_j += 1
#     index_i += 1
#
# point = keyDesc[min_index_i][0][min_index_j]
# point_x = int(point.pt[0])
# point_y = int(point.pt[1])
#
#
# patch_image = images[0][point_x:point_x+100, point_y:point_y+100]
#
# cv2.imwrite('cluster_center_3.jpg', patch_image)


histograms = []
count = 0
for image in images:
    keypoint, descriptor = sift.detectAndCompute(image, None)
    prediction = kmeans.predict(descriptor)
    histogram = np.zeros(num_clusters + 1)
    for i in prediction:
        histogram[i] += 1
    histogram = histogram / np.sum(histogram)

    histograms.append((histogram, answers[count]))
    count += 1

test_images = []
test_answers = []
count = 0

for i in glob.glob("data\\validation\\*"):
    for img in glob.glob(i+"/*.jpg"):
        n = cv2.imread(img)
        test_images.append(n)
        temp = img.split("\\")
        test_answers.append((count, temp[-2]))
        count += 1

test_predictions = []
count = 0
for img in test_images:
    keypoint, descriptor = sift.detectAndCompute(img, None)
    prediction = kmeans.predict(descriptor)
    test_histogram = np.zeros(num_clusters + 1)
    for i in prediction:
        test_histogram[i] += 1
    test_histogram = test_histogram / np.sum(test_histogram)

    test_predictions.append(kNearestNeighbors(5, histograms, test_histogram))

final_answers = [i[1] for i in test_answers]

correct_count = 0
for i in range(0, len(test_predictions)):
    if test_predictions[i] == final_answers[i]:
        correct_count += 1

print(correct_count / len(test_predictions))


#Code for Confusion Matrix

#
# confusion_matrix = {"Coast":{},"Forest":{},"Highway":{},"Kitchen":{},"Mountain":{},"Office":{},"OpenCountry":{},"Street":{},"Suburb":{},"TallBuilding":{}}
#
#
# for i in confusion_matrix:
#     confusion_matrix[i] = {"Coast":0,"Forest":0,"Highway":0,"Kitchen":0,"Mountain":0,"Office":0,"OpenCountry":0,"Street":0,"Suburb":0,"TallBuilding":0}
#
# for i in range(0,len(test_predictions)):
#      confusion_matrix[final_answers[i]][test_predictions[i]] += 1
#
# temp = {"Coast":0,"Forest":0,"Highway":0,"Kitchen":0,"Mountain":0,"Office":0,"OpenCountry":0,"Street":0,"Suburb":0,"TallBuilding":0}
#
# for i in final_answers:
#     temp[i] += 1
#
# print("            ", end=" ")
# for i in confusion_matrix:
#     print(i, end=" ")
#
# print()
# for i in confusion_matrix:
#     print(i, end="     ")
#     for j in confusion_matrix[i]:
#         val = confusion_matrix[i][j] / temp[i]
#         print("%.3f" % val, end="     ")
#     print()
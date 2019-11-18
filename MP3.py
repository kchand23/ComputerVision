import cv2
import numpy as np
import math

def houghTransform(edges, threshold,angle_res,image):

    diagonal_length = math.ceil(math.sqrt(edges.shape[0]**2 + edges.shape[1]**2))
    thetas = np.deg2rad(np.arange(-90.0, 90.0,angle_res))
    print(thetas.shape)
    acc = np.zeros((2*diagonal_length+1, len(thetas)))

    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i][j] != 0:
                for theta in range(len(thetas)):
                    rhoVal = i * np.cos(theta * np.pi / 180.0) + \
                                                     j * np.sin(theta * np.pi / 180)
                    rhoIdx = round(rhoVal) + diagonal_length
                    rhoIdx = int(rhoIdx)
                    acc[rhoIdx][theta] += 1

    lines = []

    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            if acc[i][j] > threshold/1.1:
                lines.append([i - diagonal_length, -thetas[j]])


    for line in lines:
        rho = line[0]
        theta = line[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(test,(x1,y1),(x2,y2),(0,0,255),1)

    cv2.imshow('title', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


test = cv2.imread('input.bmp')


test_gray = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
test_canny = cv2.Canny(test_gray,50,150)


houghTransform(test_canny,50,1.0,test)
# ComputerVision
Collection of all my projects related to Computer vision. 

1) Image Filters: A program intended to experiment with linear and non linear kernels on both RGB and Grayscale Images. Kernels used were Mean filter, Gaussian Filter, Sharpening Filter, Median Filter. 
                  Libraries Used were OpenCV, and NumPy
2) Edge Detection (mp2.py) : Created a Canny Edge detector, using techniques of convolution to compute the edges of any RGB Picture. 
3) Shape recognition (mp3.py) : Used Canny edge detector, and Hough Transform to detect straight lines in a picture, which can be further used to detect shapes. 
4) Color Based Segmentation (mp4.py) : Used a 2d histogram model to learn from a given set of images with HSV values, and then segment any given image based on color. 
5) Object detection based on Histogram Model (mp5.py) : Train a histogram model given a dataset of classified images. Utilized Kmeans clustering algorithm on SIFT feature descriptors to create the histograms, and then utilized K Nearest Neighbors Alogirthm to predict the label of a given image. 

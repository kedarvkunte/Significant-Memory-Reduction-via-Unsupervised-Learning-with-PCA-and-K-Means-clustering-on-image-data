# Significant Memory Reduction via Unsupervised Learning with PCA and K-Means clustering on image data
In this project, unsupervised learning is implemented with the help of python modules such as Numpy, Scipy, Sklearn, Matplotlib, PIL.

## Principal Component Analysis (PCA)
The Principal Component Analysis is an algorithm to find the directions that maximize the variance in the data. Data is projected from the actla directions is projected on the aforementioned dimensions.

## K-Means Clustering Algorithm
In this algorithm, the main aim is to form the cluster and their centroids and assigning each sample in the data to one of these clusters.

## Requirements

* Python 3.5 or higher
* Numpy
* Scipy
* Matplotlib
* PIL
* sklearn

## Dataset

In this file, there are two datasets used.
* For PCA, 100 grayscale images of size 50 X 50 pixels are used.
* For K-Means, the dataset is an image `times_sqaure.jpg` of size 400 X 400 X 3 pixels

# Analysis

The analysis of K-Means algorithm can be done with the help of following elbow plot.
![Elbow Plot](https://github.com/kedarvkunte/Significant-Memory-Reduction-via-Unsupervised-Learning-with-PCA-and-K-Means-clustering-on-image-data/blob/master/Output%20Results/Plots%20for%20K-Means%20Clustering/Elbow%20Plot.png)

K-means clustering can be a very useful image compression algorithm. 

| K  | Reconstruction error | Compression rate |
| ---: | ---: | ---: |
| 2  | 40.0753 | 0.0417 |
| 5 | 23.3516 | 0.1251 |
| 10 | 16.5217 | 0.1669 |
| 25 | 10.9738 | 0.2089 |
| 50 | 8.3434 | 0.2512 |
| 75 | 7.1580 | 0.2935 |
| 100 | 6.4359 | 0.2941 |
| 200 | 5.0760 | 0.3383 |


The mean reconstruction error using K = 25 is only about 10.97, while achieving a compression rate of 0.2.
The original image and the image with 200 colors are shown below.
####**At K = 200, the image is almost like original image, while the size is only 1/3rd of the original image thus achieving 66.66% memory reduction.**

Original Image             |  Image with K = 200
:-------------------------:|:-------------------------:
![Original Image](https://github.com/kedarvkunte/Significant-Memory-Reduction-via-Unsupervised-Learning-with-PCA-and-K-Means-clustering-on-image-data/blob/master/Output%20Results/Plots%20for%20K-Means%20Clustering/KMeans%20Original%20times_square.png)  |  ![Image with K = 200](https://github.com/kedarvkunte/Significant-Memory-Reduction-via-Unsupervised-Learning-with-PCA-and-K-Means-clustering-on-image-data/blob/master/Output%20Results/Plots%20for%20K-Means%20Clustering/KMeans%20K%20%3D%20200.png)

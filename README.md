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
| ------------- | ------------- | ---------------|
| 2  | 40.0753  | 0.04171 |



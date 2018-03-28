# Significant Memory Reduction via Unsupervised Learning with PCA and K-Means clustering on image data
In this project, unsupervised learning is implemented with the help of python modules such as Numpy, Scipy, Sklearn, Matplotlib, PIL.
The Principal Component Analysis (PCA) is an algorithm to find the directions that maximize the variance in the data. Data is projected from the actla directions is projected on the aforementioned dimensions. In K-Means Clustering Algorithm algorithm, the main aim is to form the cluster and their centroids and assigning each sample in the data to one of these clusters.

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

## Analysis

Original Image             |  Image with K = 200
:-------------------------:|:-------------------------:
![Original Image](https://github.com/kedarvkunte/Significant-Memory-Reduction-via-Unsupervised-Learning-with-PCA-and-K-Means-clustering-on-image-data/blob/master/Output%20Results/Plots%20for%20K-Means%20Clustering/KMeans%20Original%20times_square.png)  |  ![Image with K = 200](https://github.com/kedarvkunte/Significant-Memory-Reduction-via-Unsupervised-Learning-with-PCA-and-K-Means-clustering-on-image-data/blob/master/Output%20Results/Plots%20for%20K-Means%20Clustering/KMeans%20K%20%3D%20200.png)


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
| **200** |**5.0760** | **0.3383** |


The mean reconstruction error using K = 25 is only about 10.97, while achieving a compression rate of 0.2.
The original image and the image with 200 colors are shown above.
**At K = 200, the image is almost like original image, while the size is only 1/3rd of the original image thus achieving 66.66% memory reduction.**

For PCA, similar analysis can be performed with the help of reconstruction error and compression rate. PCA face gridplot is shown below for face 1.
![Face grid plot](https://github.com/kedarvkunte/Significant-Memory-Reduction-via-Unsupervised-Learning-with-PCA-and-K-Means-clustering-on-image-data/blob/master/Output%20Results/Plots%20for%20PCA/PCA%20Face1%20Gridplot.png)

The table for k (where k eigen vectors are chosen to maximize the variance using k largest eigen values), reconstruction error and compression rate is shown as below:

| K  | Reconstruction error | Compression rate |
| ---: | ---: | ---: |
| 3  | 31.5818 | 0.031 |
| 5 | 28.4036 | 0.052 |
| 10 | 22.9751 | 0.104  |
| 30 | 13.4085 | 0.312 |
| 50 | 8.1921  | 0.520 |
| 100 | 2.4407e-12 | 1.040 |
| 150 | 2.4383e-12 | 1.560 |
| 300  | 2.44711e-12 | 3.120 |

The PCA grid plot shows K principal components. At K = 50, Reconstruction error is around 8.1921, with compression rate almost equal 0.5 **thus achieving 50% memory reduction with still displaying many features.**







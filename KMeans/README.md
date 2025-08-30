# K-Means Clustering from Scratch

## Introduction
K-Means clustering is an unsupervised machine learning algorithm. 
It aims to partition data into a specified number (K) of clusters 
by iteratively assigning data points to the nearest centroid 
and then recalculating centroids until cluster assignments stabilize.

I built a K-Means cluster from scratch by numpy to understand the key concepts and algorithm. 
This cluster is test by the [Iris Species](https://www.kaggle.com/datasets/uciml/iris) dataset. 
If this algorithm run correctly, each cluster should be corresponding to a species of Iris.

I learned this algorithm by this [notebook](https://www.kaggle.com/code/fareselmenshawii/kmeans-from-scratch/notebook).

## Algorithm
### Initialization
The first step of K-Means clustering is initializing the centroids. 
We just select $K$ data points from dataset randomly.
These centroids are data points in our dataset that will act as the centers of our clusters.

### Assignment of Points to Centroids
The second step is to assign data points to centroids. 
For each data point in our dataset, we determine which centroid it is closest to. 
This is done by measuring the distance between the data point and each centroid and selecting the centroid with the smallest distance as the closest one.

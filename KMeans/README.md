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
For each data point in our dataset, we compute the distance between the data point and each centroid:
```math
d_i = \Vert x-\mu_i \Vert \equiv \sqrt{|x-\mu_i|^2}
```
where $x$ is the data point and $\mu_i (i=1,2,...,K)$ is the centroid point. 
We will have $K$ distance, and then we assign the data point to the centroid with the smallest distance. 

### Computation of Means
After assigning data points to their closest centroids, we calculate the mean of all data points within each cluster. 
This mean becomes the new centroid for that cluster:
```math
\mu_i = \frac{1}{|S_i|}\sum_{x\in S_i}x
```
where $S_i$ is a set that contains data points in the $i$-th cluster, and $|S_i|$ represents its size. 

### Training
We repeat the assignment and computation of centroids `max_iteration` times. 
If the centroids does not change any more, we say the algorithm is convergence and stop running.

## Result
Fig. 1 shows the SepalLength vs PetalLength scatter plot by using the dataset. 
Three true species of iris are labeled by different color. 
These species can be distinguished clearly by this plot.

![Three iris species](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/blob/main/KMeans/true.png)
*Fig. 1. Three iris species*

Fig. 2 shows the result of K-Means clustering. 
The algorithm stops after 12 iterations. 
The centroids are displayed by the red plus symbols. 
We can see some yellow points are mislabeled to dark green, but this algorithm works principally.

![K-Means result](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/blob/main/KMeans/kmeans.png)
*Fig. 2. K-Means Result*

## Conclusion
I built a K-Means clustering from scratch. This is a simple algorithm and easy to understand. 
This algorithm is test by the iris dataset and works well.

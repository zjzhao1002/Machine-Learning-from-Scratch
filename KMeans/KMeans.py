import numpy as np

class KMeans():
    def __init__(self, K: int):
        """
        The constructor for KMeans class.
        Args:
            K: The number of clusters.
        """
        if K <= 0:
            raise Exception("The parameter K should be a integer greater than 0.")
        self.K = K

    def initialize_centroids(self, data: np.ndarray):
        """
        This function initializes the centroids for each cluster by selecting K points randomly from the dataset.
        Args: 
            data: The input dataset
        """
        if data.shape[0] < self.K:
            raise Exception("The number of data should be greater the number of clusters.")
        
        np.random.seed(1)
        n_samples = data.shape[0]

        shuffled_indices = np.random.permutation(np.arange(n_samples))

        ccntroid_indices = shuffled_indices[:self.K]
        ccntroids = data[ccntroid_indices]
        return ccntroids
    
    def assign_points(self, data: np.ndarray, centroids: np.ndarray):
        """
        This function caculate the distance between centroids and other points in the dataset, 
        and assigns each point in the dataset to the nearest centroid. 
        Args: 
            data: The input dataset
            centroids: The current centroids
        """
        data = np.expand_dims(data, axis=1)
        distance = np.linalg.norm((data-centroids), axis=-1)
        points = np.argmin(distance, axis=1)
        return points
    
    def compute_mean(self, data: np.ndarray, points: np.ndarray):
        """
        This function computes the mean of the points assigned to each centroid. 
        Args:
            data: The input data
            points: An array containing the index of the centroid for each point
        """
        centroids = np.zeros((self.K, data.shape[1]))

        for i in range(self.K):
            centroid_mean = data[points == i].mean(axis=0)
            centroids[i] = centroid_mean

        return centroids
    
    def fit(self, data: np.ndarray, max_iterations: int=10):
        """
        This function clusters the dataset using the K-Means algorithm.
        Args:
            data: The dataset to be clustered
            max_iterations: The max interations to perform run the algorithm.
        """
        controids = self.initialize_centroids(data)
        old_controids = np.zeros((self.K, data.shape[1]))

        i=0
        while i < max_iterations and not (controids==old_controids).all():
            old_controids = controids
            points = self.assign_points(data, controids)
            controids = self.compute_mean(data, points)
            i+=1

        print(f"Finish after {i} itertaions.")

        return controids, points

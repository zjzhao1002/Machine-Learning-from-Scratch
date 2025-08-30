import pandas as pd
import numpy as np
from KMeans import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("Iris.csv")

print(df.columns)
X = df.drop(["Id", "Species"], axis=1).values
y = df["Species"].replace(["Iris-setosa", "Iris-versicolor", "Iris-virginica"], [0, 1, 2])

# These codes make the scatter plot by the dataset and labels. 
# plt.scatter(x=X[:, 0], y=X[:, 2], c=y)
# plt.xlabel("SepalLength (cm)")
# plt.ylabel("PetalLength (cm)")
# plt.show()

kmeans = KMeans(3)
centroids, points = kmeans.fit(X, max_iterations=100)

# These codes make the scatter plot to show the result of KMeans clustering.
# plt.scatter(x=X[:, 0], y=X[:, 2], c=points)
# plt.scatter(x=centroids[:, 0], y=centroids[:, 2], marker="P", c="red")
# plt.xlabel("SepalLength (cm)")
# plt.ylabel("PetalLength (cm)")
# plt.show()
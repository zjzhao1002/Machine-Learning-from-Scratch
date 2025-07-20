import network
import pandas as pd
import numpy as np

data = pd.read_csv("train.csv")
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_val = data[0:1000].T
Y_val = data_val[0]
X_val = data_val[1:n]
X_val = X_val / 255.

data_train = data[1000: m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.

net = network.Network([784, 10, 10])
net.train(x=X_train, y=Y_train, epochs=100, eta=1)

predictions = net.predict(X_val)
accuracy = net.get_accuracy(predictions, Y_val)
print(accuracy)
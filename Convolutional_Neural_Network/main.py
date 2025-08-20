from idx2numpy import convert_from_file
from cnn import CNN
import numpy as np

X_train = convert_from_file("train-images-idx3-ubyte")
y_train = convert_from_file("train-labels-idx1-ubyte")
X_val = convert_from_file("t10k-images-idx3-ubyte")
y_val = convert_from_file("t10k-labels-idx1-ubyte")
batch_size, height, width = X_train.shape 
channels = 1
X_train = X_train.reshape((batch_size, channels, height, width)) # Reshape data by adding a dimension for channel
X_train = X_train / 255
batch_size, height, width = X_val.shape
X_val = X_val.reshape((batch_size, channels, height, width)) # Reshape data by adding a dimension for channel
X_val = X_val / 255

network = CNN(in_channels=1, out_channels=1, linear_input_size=196, linear_output_size=10)

network.train(X_train, y_train, epochs=10, eta=0.01)

predictions = network.predict(X_train)
accuracy = network.get_accuracy(predictions, y_train)
print("The accuracy for the training dataset: ", accuracy)

predictions = network.predict(X_val)
accuracy = network.get_accuracy(predictions, y_val)
print("The accuracy for the validation dataset: ", accuracy)
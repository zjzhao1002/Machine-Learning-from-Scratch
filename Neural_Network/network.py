import numpy as np

def ReLU(x):
    return np.maximum(x, 0)

def ReLU_deriv(x):
    return x > 0

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def one_hot(x):
    one_hot_x = np.zeros((x.size, x.max()+1))
    one_hot_x[np.arange(x.size), x] = 1
    one_hot_x = one_hot_x.T
    return one_hot_x

class Network(object):

    def __init__(self, layer_sizes):
        """
        'layer_sizes' is a list contains the number of neurons in 
        the respective layers of the networks. The length of it is 
        the number of layers.
        """
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.biases = []
        self.weights = []

        for i in range(1, self.num_layers):
            j = self.layer_sizes[i]
            k = self.layer_sizes[i-1]
            self.biases.append(np.random.rand(j, 1)-0.5)
            self.weights.append(np.random.rand(j, k)-0.5)

    def feedforward(self, a):
        """
        Calculate all weighted input z and activations in the network.
        """
        a_list = [a]
        z_list = []
        for i in range(self.num_layers-2):
            z = np.dot(self.weights[i], a) + self.biases[i]
            a = ReLU(z)
            a_list.append(a)
            z_list.append(z)
        z = np.dot(self.weights[self.num_layers-2], a) + self.biases[self.num_layers-2]
        a = softmax(z)
        a_list.append(a)
        z_list.append(z)
        return z_list, a_list
    
    def backpropagation(self, x, y):
        """
        Perform the backpropagation. It returns the gradient descent 
        for the lost function. 'nabla_b' is the partial derivatives of 
        the loss function with respect to biases, while 'nabla_w' is the 
        partial derivatives of the loss function with respect to weights.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        zs, activations = self.feedforward(activation)
        delta = activations[-1] - y # The loss function is assumed to be cross entropy loss
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].T, delta) * ReLU_deriv(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
        return (nabla_b, nabla_w)
    
    def update_parameters(self, x, y, eta):
        """
        Update the network's weights and biases by applying gradient 
        descent using backpropagation. 
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        m, n = x.shape
        my, _ = y.shape
        for i in range(n):
            xb = x[:, i].reshape(m,1)
            yb = y[:, i].reshape(my,1)
            delta_nabla_b, delta_nabla_w = self.backpropagation(xb, yb)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.biases = [b-(eta/n)*nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w-(eta/n)*nw for w, nw in zip(self.weights, nabla_w)]

    def train(self, x, y, epochs, eta):
        """
        Train the network. '(x, y)' represent the training inputs and 
        desired output. 'epochs' is the number of iterations. 'eta' 
        represents the learning rate. 
        """
        one_hot_y = one_hot(y)
        for i in range(epochs):
            self.update_parameters(x, one_hot_y, eta)
            if (i+1) % 50 == 0:
                print(f"Running {i+1} epoch")
                predictions = self.predict(x)
                accuracy = self.get_accuracy(predictions, y)
                print("Accuracy: ", accuracy)

    def predict(self, x):
        """
        Give the predictions of the network with input x.
        """
        z, a = self.feedforward(x)
        return np.argmax(a[-1], 0)
    
    def get_accuracy(self, predictions, y):
        """
        Calculate the accuracy.
        """
        return np.sum(predictions == y) / y.size
import numpy as np
from scipy.signal import correlate2d

def ReLU(x):
    """
    The ReLU function.
    """
    return np.maximum(x, 0)

def ReLU_deriv(x):
    """
    The derivative of the ReLU function.
    """
    return x > 0

def softmax(x):
    """
    The softmax function.
    """
    return np.exp(x) / sum(np.exp(x))

def one_hot(x):
    """
    One hot encoding.
    """
    one_hot_x = np.zeros((x.size, x.max()+1))
    one_hot_x[np.arange(x.size), x] = 1
    one_hot_x = one_hot_x.T
    return one_hot_x

class conv2d():
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, padding: int=1):
        """
        The constructor for the convolutional leayer. For simplicity, stride, dilation and biases are not considered.
        Args:
            in_channels: The number of channels of the input data. 
            out_channels: The number of channels of the output.
            kernel_size: The size of the kernel. A kernel with height=width=kernel_size is generated
            padding: The size of the padding. 
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)

    def forward(self, data: np.ndarray):
        """
        Forward propagation of the convolutional layer.
        Args:
            data: A single data sample
        """
        if self.padding > 0:
            data = np.pad(data, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')

        self.data = data
        channels_in, input_height, input_width = data.shape
        channels_out, channels_in, kernel_height, kernel_width = self.kernels.shape

        output_height = np.floor(1 + (input_height- kernel_height)).astype(int)
        output_width = np.floor(1 + (input_width - kernel_width)).astype(int) 

        out = np.zeros((channels_out, output_height, output_width), dtype=np.float32)

        for i in range(channels_out):
            for j in range(channels_in):
                out[i] = correlate2d(data[j], self.kernels[i, j], mode='valid')

        # The following code can do the same things as correlate2d, but much slower.
        #
        # for c_y in range(output_height):
        #     for c_x in range(output_width):
        #         for c_channel_out in range(channels_out):
        #             for c_channel_in in range(channels_in):
        #                 for c_kernel_y in range(kernel_height):
        #                     for c_kernel_x in range(kernel_width):
        #                         this_pixel_value = data[c_channel_in, c_y+c_kernel_y, c_x+c_kernel_x]
        #                         this_weight = self.kernels[c_channel_out, c_channel_in, c_kernel_y, c_kernel_x]
        #                         out[c_channel_out, c_y, c_x] += np.sum(this_pixel_value*this_weight)

        return out
    
    def backward(self, delta: np.ndarray, eta: float):
        """
        The backward propagation of the convolutional layer. The weights of the kernels are also updated. 
        Args:
            delta: The gradient of the next layer of this model
            eta: The learning rate
        """
        delta_data = np.zeros(self.data.shape)
        delta_kernels = np.zeros(self.kernels.shape)

        for i in range(self.out_channels):
            for j in range(self.in_channels):
                delta_kernels[i, j] = correlate2d(self.data[j], delta[i], mode='valid')
                delta_data[j] += correlate2d(delta[i], self.kernels[i, j], mode='full')

        self.kernels = self.kernels - eta*delta_kernels

        return delta_data

    
class maxpool():
    def __init__(self, pool_size: int=2):
        """
        The constructor of the max pooling layer. 
        Args:
            pool_size: The size of the window to take a max over. pool_size is used for the height and width.
        """
        self.pool_size = pool_size

    def forward(self, data: np.ndarray):
        """
        The forward propagation of the max pooling layer. 
        Args:
            data: A single data sample.
        """
        self.data = data
        channels_in, input_height, input_width = data.shape

        output_height = np.floor(input_height / self.pool_size).astype(int)
        output_width = np.floor(input_width / self.pool_size).astype(int)

        out = np.zeros((channels_in, output_height, output_width))

        for c in range(channels_in):
            for i in range(output_height):
                for j in range(output_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size

                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size

                    patch = data[c, start_i:end_i, start_j:end_j]
                    out[c, i, j] = np.max(patch)

        return out     

    def backward(self, delta: np.ndarray):
        """
        The backward propagation of the max pooling layer.
        Args:
            delta: The gradient of the next layer in this model.
        """
        delta_new = np.zeros(self.data.shape)

        channels_in, input_height, input_width = self.data.shape

        output_height = np.floor(input_height / self.pool_size).astype(int)
        output_width = np.floor(input_width / self.pool_size).astype(int)

        for c in range(channels_in):
            for i in range(output_height):
                for j in range(output_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size

                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size

                    patch = self.data[c, start_i:end_i, start_j:end_j]
                    mask = patch == np.max(patch)
                    delta_new[c, start_i:end_i, start_j:end_j] = delta[c, i, j] * mask

        return delta_new

class flatten():
    def __init__(self):
        """
        The contructor of the flatten layer. 
        """
        pass

    def forward(self, data: np.ndarray):
        """
        The forward propagation of the flatten layer. It actually just reshapes the input data to one dimension.
        Args:
            data: A single data sample.
        """
        self.input_shape = data.shape
        out = data.flatten()
        out = out.reshape(out.shape[0], 1)
        return out
    
    def backward(self, delta: np.ndarray):
        """
        The backward propagation of the flatten layer. It actually just reshapes the one dimension data to the input shape.
        Args:
            data: A single data sample.
        """
        out = delta.reshape(self.input_shape)
        return out

class linear():
    def __init__(self, input_size: int, output_size: int):
        """
        The constructor of the linear layer. 
        Args:
            input_size: The size of the input sample.
            output_size: The size of the output sample.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)

    def forward(self, data: np.ndarray):
        """
        The forward propagation of the linear layer.
        Args:
            data: A single data sample.
        """
        self.data = data
        return np.dot(self.weights, data) + self.biases
    
    def backward(self, delta: np.ndarray, eta: float):
        """
        The backward propagation of the linear layer.
        Args:
            delta: The gradient of the next layer in the model.
            eta: The learning rate.
        """
        nabla_b = np.zeros(self.biases.shape)
        nabla_w = np.zeros(self.weights.shape)

        nabla_b = delta
        nabla_w = np.dot(delta, self.data.T)

        delta = np.dot(self.weights.T, delta)

        self.biases = self.biases - eta*nabla_b
        self.weights = self.weights - eta*nabla_w

        return delta
    
class CNN():
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int=3, 
                 padding: int=1,
                 pool_size: int=2, 
                 linear_input_size: int=4,
                 linear_output_size: int=2):
        """
        The constructor of the Convolutional Neural Network (CNN) class. This network contains a convolutional layer, 
        a max pooling layer, a flatten layer and a linear layer.
        Args:
            in_channels: The number of channels of the input data. 
            out_channels: The number of channels of the output.
            kernel_size: The size of the kernel. A kernel with height=width=kernel_size is generated.
            padding: The size of the padding. 
            pool_size: The size of the window to take a max over. pool_size is used for the height and width.
            linear_input_size: The size of the input sample for the linear layer.
            linear_output_size: The size of the output sample for the linear layer. It is also the number of classes for classification.            
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.pool_size = pool_size
        self.linear_input_size = linear_input_size
        self.linear_output_size = linear_output_size

        self.conv = conv2d(self.in_channels, self.out_channels, self.kernel_size, self.padding)
        self.pool = maxpool(self.pool_size)
        self.flatten_layer = flatten()
        self.linear_layer = linear(linear_input_size, linear_output_size)

    def forward(self, X: np.ndarray):
        """
        Forward propagation of the network. Data will pass through a convolutional layer, 
        a maxpooling layer, a flatten layer, and a linear layer.
        Args:
            X: A single data sample.
        """
        channels_in, input_height, input_width = X.shape 
        if channels_in != self.in_channels:
            raise Exception("Your data have different input channels to the convolutional layer.")

        self.conv_out = self.conv.forward(X)
        relu_out = ReLU(self.conv_out)
        pool_out = self.pool.forward(relu_out)
        flatten_out = self.flatten_layer.forward(pool_out)

        input_size = flatten_out.shape[0]
        if input_size != self.linear_input_size:
            raise Exception("Your data have different input size to the linear layer. Please check the input parameters.")
        
        linear_out = self.linear_layer.forward(flatten_out)
        final_out = softmax(linear_out)

        return final_out
    
    def backward(self, delta: np.ndarray, eta: float):
        """
        Backward propagation of the network. The gradient of the linear layer, flatten layer, max pooling layer and convolutional layer are calculated.
        Args:
            delta: The derivative of the loss function with respect to the weighted input to the final activation function.
            eta: The learning rate. 
        """
        linear_delta = self.linear_layer.backward(delta, eta)
        flatten_delta = self.flatten_layer.backward(linear_delta)
        pool_delta = self.pool.backward(flatten_delta)
        relu_delta = pool_delta * ReLU_deriv(self.conv_out)
        conv_delta = self.conv.backward(relu_delta, eta)

        return conv_delta
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int=5, eta: float=0.1):
        """
        This funcion is used to train the network.
        Args:
            X: The input features for training.
            y: The target values.
            epochs: The number of iterations.
            eta: The learning rate.
        """
        num_samples = X.shape[0]
        one_hot_y = one_hot(y)
        my = one_hot_y.shape[0]
        for epoch in range(epochs):
            for i in range(num_samples):
                forward_out = self.forward(X[i])
                this_y = one_hot_y[:, i].reshape(my, 1)
                delta = forward_out - this_y # The loss function is assumed to be cross entropy loss.
                delta_out = self.backward(delta, eta)

            # if (epoch+1) % 10 == 0:
            predictions = self.predict(X)
            # print(predictions)
            accuracy = self.get_accuracy(predictions, y)
            print(f"Epoch {epoch+1}/{epochs}, Accuracy: {accuracy}")

    def predict(self, X: np.ndarray):
        """
        Give the predictions of the network with input X.
        Args: 
            X: The data to make predictions for.
        """
        num_samples = X.shape[0]
        predictions = np.zeros((self.linear_output_size, num_samples))
        for i in range(num_samples):
            network_out = self.forward(X[i])
            predictions[:, i] = network_out.reshape(self.linear_output_size)
        predictions = np.array(predictions)
        # print(predictions.shape)
        return np.argmax(predictions, 0)
    
    def get_accuracy(self, predictions: np.ndarray, y: np.ndarray):
        """
        This function calculate the accuracy of a classification model.
        Args:
            predictons: The predicted labels for each data point.
            y: The true labels for each data point.
        Returns:
            The accuracy of the model.
        """
        return np.sum(predictions==y) / y.size
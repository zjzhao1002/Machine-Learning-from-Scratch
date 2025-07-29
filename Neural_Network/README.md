# Neural Network from Scratch

## Motivation

In machine learning, a neural network (also artificial neural network or neural net, abbreviated ANN or NN)
is a computational model inspired by the structure and functions of biological neural networks.

To understand the key concepts and maths of neural network, I built a neural network from scratch by using **numpy** only.
The purpose of this network is to recognize handwritten digits in the 
[MNIST Dataset](https://www.kaggle.com/datasets/animatronbot/mnist-digit-recognizer).
I mainly follow this [book](http://neuralnetworksanddeeplearning.com/).

## Algorithm
A typical neural network is looked like:

![image from wikipedia](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/blob/main/Neural_Network/Colored_neural_network.svg)

The circles in this image are called neurons. Neurons are organized into layers: 
input layer (receives initial data), hidden layers (perform computations), and output layer (provides the final result). 
It is allowed to have multiple hidden layers, which is called Deep Learning.
Neurons are connected by weighted links, representing the strength of the connection. 
These weights are adjusted during the learning process. 

### Neuron
A neuron takes n inputs $x=(x_1, x_2, ..., x_n)$ and produce a single output 
```math
z = W\cdot x+b,
```
where $W$ is the weight matrix to connect this layer to next layer, and $b$ is the bias. 
$W$ and $b$ are learnable parameters. A neural network is trained to optimize them.

### Activation Function
The output $z$ is just a linear transformation of the inputs. We can only use this network to learn linear relationship. 
To learn complex patterns in data, nonlinearity is introduced by activation functions.
In this work, I have introduced ReLU function for hidden layer:
```math
\sigma(x) = \begin{cases}
0 & \text{if } x\leq 0 \\
x & \text{if } x>0 
\end{cases}
```
Since this is a multi-class classification problem, the activation function for output layer is the softmax function:
```math
\sigma(x)_i = \frac{e^{x_i}}{\sum^{C}_{j=1}e^x_j},
```
$C$ is the number of classes. In our case, $C=10$.
So the final output of a neuron is 
```math
a = \sigma(z) = \sigma\left( W\cdot x+b \right)
```

### Feedforward
Data are input to the input layer and then pass through the hidden layer. 
We use previous equations to calculate activation $a$ for each layer. 
Finally, the activation from the last hidden layer is passed to output layer to make predictions.
The activation of the output layer gives the predictions.
This process is called feedforward.

### Backpropagation
The parameters $W$ and $b$ are learned by gradient descent. The goal of this method is minimizing the loss function (cross entropy loss):
```math
L(a,y) =  -\sum^C_{j=1}y_j\log{a_j}
```
To do that, we first calculate the error $\delta_j^l$ of neuron $j$ in layer l:
```math
\delta_j^l\equiv \frac{\partial L}{\partial z_j^l}
```
For the output layer $L$, we have
```math
\delta_j^L = \frac{\partial L}{\partial z_j^L} 
= \sum_k\frac{\partial L}{\partial a_k^L}\frac{\partial a_k^L}{\partial z_j^L}
```
With the definition of loss fuction, we have
```math
\frac{\partial L}{\partial a_j^L} = -\frac{y_j}{a_j}
```
For the second part, we just need the derivative of the softmax function:
```math
\frac{\partial a_k^L}{\partial z_j^L} = a_k^L(\delta_{kj}-a_j^L),
```
where $\delta_{kj}$ is the Kronecker delta. Combining these results, we get
```math
\delta_j^L = -y_j + a_j\sum_k y_k
```
In this work, $y$ is one hot encoded, the summation of $y_k$ is equal to one. 
The final result is 
```math
\delta_j^L = a_j - y_j
```
With this result, we can calculate other $\delta^l$: 
```math
\delta^l = \left( (W^{l+1})^T\delta^{l+1} \right)\odot \sigma^\prime(z^l)
```
$\sigma^\prime$ is the derivative of the ReLU function, and $\odot$ is the elementwise product.
Now we can calculate these derivatives:
```math
\begin{split}
\frac{\partial L}{\partial b^l_j} = \delta^l_j \\
\frac{\partial L}{\partial W^l_{jk}} = a^{l-1}_k\delta^l_j
\end{split}
```
The proof of these equations can be found [here](http://neuralnetworksanddeeplearning.com/). 
With these equations, the parameters can be updated by
```math
\begin{split}
W^l \to W^l - \frac{\eta}{n}\sum_x\delta^{x,l}(a^{x,l-1})^T, \\
b^l \to b^l - \frac{\eta}{n}\sum_x\delta^{x,l},
\end{split}
```
where $x$ represents a data sample for training, $n$ is the number of $x$, and $\eta$ is the learning rate. 

### Training
After updating the parameters, we can feedforward again by new parameters. 
We should has a smaller $L$. Repeating the feedforward and backpropagation `epochs` times, 
we can minimize the loss function and optimize the parameters.

### Predictions
After the optimization of the parameters, we can input some data points to the network. 
The feedforward process can use the optimal parameters to give the prediction for each data point.
With the `epochs=100` and `eta=1` (learning rate), the network can reach around 85% accuracy for both training and test dataset. 
This is an acceptable result for such naive from scratch implementation.

## Conclusion
I have build a neural network from scratch to understand the maths and key concepts. 
It is a beautiful algorithm and the basis of modern machine learning models. 
This also means many things can be improved. 

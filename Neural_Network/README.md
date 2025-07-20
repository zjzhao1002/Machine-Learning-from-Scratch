# Neural Network from Scratch

## Motivation

In machine learning, a neural network (also artificial neural network or neural net, abbreviated ANN or NN)
is a computational model inspired by the structure and functions of biological neural networks.

To understand the key concepts and maths of neural network, I built a neural network from scratch by using **numpy** only.
The purpose of this network is to recognize handwritten digits in the 
[MNIST Dataset](https://www.kaggle.com/datasets/animatronbot/mnist-digit-recognizer).
I mainly follow this [book](http://neuralnetworksanddeeplearning.com/).

## Maths
A typical neural network is looked like:

![image from wikipedia](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/blob/main/Neural_Network/Colored_neural_network.svg)

The circles in this image are called neurons. Neurons are organized into layers: 
input layer (receives initial data), hidden layers (perform computations), and output layer (provides the final result). 
It is allowed to have multiple hidden layers, which is called Deep Learning.
Neurons are connected by weighted links, representing the strength of the connection. 
These weights are adjusted during the learning process. 

### Neuron
A neuron takes n inputs $\hat{x}=(x_1, x_2, ..., x_n)$ and produce a single output 
```math
z = W\cdot \hat{x}+\hat{b},
```
where $W$ is the weight matrix to connect this layer to next layer, and $\hat{b}$ is the bias. 
$W$ and $\hat{b}$ are learnable parameters. A neural network is trained to optimize them.

### Activation Function
The output $z$ is just a linear transformation of the inputs. We can only use this network to learn linear relationship. 
To learn complex patterns in data, nonlinearity is introduced by activation functions.
In this work, I have introduced ReLU function for hidden layer:
```math
\sigma(x) = \begin{cases}
0 & \text{if} x\leq 0 \\
x & \text{if} x>0 
\end{cases}
```
Since this is a multi-class classification problem, the activation function for output layer is the softmax function:
```math
\sigma(x)_i = \frac{e^{x_i}}{\sum^{C}_{j=1}e^x_j},
```
$C$ is the number of classes. In our case, $C=10$.

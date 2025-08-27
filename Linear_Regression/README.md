# Linear Regression

## Introduction

Linear regression is a simple machine learning model. 
It is useful in predicting a continuous variable by one or more indepedent variables.

I built a linear regression model from scratch by **numpy** to understand the key concepts and algorithm.
This model is test by the [Student Performance](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression/) dataset. 
The dataset aims to provide insights into the relationship between the independent variables and the performance index.

I learned linear regression by this [article](https://medium.com/analytics-vidhya/multiple-linear-regression-from-scratch-using-python-db9368859f).

## Algorithm

A linear regression model assumes that the relationship between the dependent variable $y$ 
and the vector of variables $x=(x_1, x_2, ..., x_N)$ is linear. 
So we have function:
```math
y = b + w_1x_1 + w_2x_2 + ... + w_Nx_N = b + x^Tw
```
where $b$ is the bias and $w=(w_1, w_2, ..., w_N)$ is the vector of weight. 

We have to train the model to optimize these weights and bias. 

### Forward Pass
The forward pass is just a step where we compute the $y$ for the input data $x$ by the current weights and biases. 
It is essentially applying our model to the input data. 
We use $f(x) = b + x^Tw$ to represent the result.

### Loss function
The loss function (or cost function) is a function to measure how well our model is performing. 
For the linear regression model, we use the following function:
```math
L(b, w) = \frac{1}{2m}\sum^m_{j=1}\left( f(x) - y \right)^2 
```
where $m$ is the total number of data samples.

### Backward pass
We use the gradient descent algorithm to minimize the loss function. 
This algorithm works by iteratively updating the parameters.

First, we compute the derivatives with respect to bias and weights:
```math
\begin{split}
\frac{\partial L}{\partial b} = \frac{1}{m}\sum^m_{j=1}\left( f(x) - y \right) \\
\frac{\partial L}{\partial w_i} = \frac{1}{m}\sum^m_{j=1}\left( f(x) - y \right)x_i
\end{split}
```

Second, we update these parameters by 
```math
\begin{split}
b \to b - \eta \frac{\partial L}{\partial b} \\
w_i \to w_i - \eta \frac{\partial L}{\partial w_i}
\end{split}
```
where $\eta$ is the learning rate.

By iteratively performing the forward pass, computing the loss function, performing the backward pass, and updating the parameters, 
the model learns to make better predictions. 

This plot show the 

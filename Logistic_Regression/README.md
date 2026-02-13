# Logistic Regression from Scratch

## Motivation 

Logistic regression is a supervised machine learning algorithm used for binary classification, 
predicting the probability of a categorical dependent variable (e.g., 0/1, yes/no) based on independent variables.
This model is simple. It uses a sigmoid function to map predictions to a probability between 0 and 1, 
so it is useful in binary classification. 

I built a linear regression model from scratch by **numpy** to understand the key concepts and algorithm. 
The model is test by the [500 Person Gender-Height-Weight-Body Mass Index](https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex) dataset. 
The purpose of this classifier is to predict if a person is obese.

## Algorithm

As mentioned above, a logistic regression model make predictions with probabilities between 0 and 1. 
We can use $$p(X)$$ represent the probability of the input $$X$$, and write:
```math
p(X) = f(W\cdot X + b),
```
where $$f(z)$$ is the sigmoid function:
```math
f(z) = \frac{1}{1+e^{-z}}
```
The sigmoid function is the mathematical bridge that turns a linear regression model into a classification model. 
It promises the predictions are probabilities between 0 and 1. 
It also introduces nonlinearity so we can calculate the gradient. 
The derivative of sigmoid function is simple: 
```math
\frac{df(z)}{dz} = f(z)(1-f(z))
```

For the logistic regression model, the loss function is cross entropy loss:
```math
L(W, b) = -\frac{1}{m} \sum^{m-1}_{i=0}\left[ y_i\log{p(X_i)} + (1-y_i)\log{(1-p(X_i))} \right],
```
where $$m$$ is the total number of data samples. This gives the gradient:
```math
\begin{aligned}
  \frac{\partial L}{\partial W} &=& \frac{1}{m} \sum^{m-1}_{i=0}(p(X_i)-y_i)X_i \\
  \frac{\partial L}{\partial b} &=& \frac{1}{m} \sum^{m-1}_{i=0}(p(X_i)-y_i)
\end{aligned}
```
So the parameters can be updated by the gradient decent method:
```math
\begin{aligned}
W &\to& W - \eta \frac{\partial L}{\partial W}, \\
b &\to& b - \eta \frac{\partial L}{\partial b},
\end{aligned}
```
where $$\eta$$ is the `learning rate`. We update these parameters `iterations` times to minimize the loss function. 

## Result
With `learning_rate=0.001` and `iterations=100`, I got cross entropy loss $$L=0.1724$$ finally. 
The accuracy reached $$92\%$$ for the training dataset and $$95\%$$ for the test dataset. 
We can say this model works well. 

## Conclusion
Logistic Regression is a simple machine learning algorithm. 
I built it from scratch to classify the [500 Person Gender-Height-Weight-Body Mass Index](https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex) dataset. 
Many concepts of this model are also applied to other machine learning models. 
This is a good practice for beginner.

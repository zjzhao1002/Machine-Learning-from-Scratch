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

As mentioned above, a logistic regression model make predictions with a probability between 0 and 1. 
We can use $$p(x)$$ represent the probability of the input $$x$$, and write:
```math
p(x) = f(W\cdot x + b),
```
where $$f(z)$$ is the sigmoid function:
```math
f(z) = \frac{1}{1+e^{-z}}
```

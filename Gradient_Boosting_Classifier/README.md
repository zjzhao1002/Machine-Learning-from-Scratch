# Gradient Boosting Classifier from Scratch

## Motivation
Gradient boosting is a machine learning technique used for regression and classification.
As a boosting technique, it gives a prediction model in the form of an ensemble of a sequence of weak models, 
which are typically simple decision trees.
[Gradient boosting for regression](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/Gradient_Boosting_Regressor) is easy to build, 
but the classifier is more complicated. 
In this project, I built a gradient boosting classifier from scratch to understand the maths and concepts. 
This model is test by the [Iris Species](https://www.kaggle.com/datasets/uciml/iris) dataset. 

I learned the gradient boosting classifier from this 
[article](https://randomrealizations.com/posts/gradient-boosting-multi-class-classification-from-scratch/) and 
this [notebook](https://www.kaggle.com/code/egazakharenko/gradient-boosting-from-scratch-full-tutorial/notebook). 
In addition, this [video](https://youtu.be/StWY5QWMXCw?si=1h6ueHUvD_JMprwV) gives a detail explanation of gradient boosting for classification.

## Algorithm
For a classification problem, the target variable in the training dataset should be integer encoded. 
Considering the dataset has $$K$$ classes, so these classes are mapped to the integers $$0,1,...,K-1$$.
To apply the gradient boosting to a classification problem, we have to convert this problem to predict some continuous variables, 
the probabilities for these classes. 
So we need a way to ensure that the model output is a valid probability mass function, 
i.e. each probability is in (0, 1) and the $$K$$ class probabilities sum to 1.

As in the [logistic regression](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/Logistic_Regression), 
we can build a model to make a prediction with real numbers, then using some function to transform these numbers to probabilities. 
Since we are going to classify $$K$$ classes, we need $$K$$ different models, one for each class.
In a multi-class problem, softmax function is used for this purpose. 
Considering the $$K$$ outputs are $$F_1(X),F_2(X),...,F_{K}(X)$$
and the corresponding probability mass function are $$p_1(X),p_2(X),...,p_{K}(X)$$, 
the softmax function gives:
```math
p_k(X) = \frac{e^{F_k(X)}}{\sum^K_{l=1}e^{F_k(X)}}
```

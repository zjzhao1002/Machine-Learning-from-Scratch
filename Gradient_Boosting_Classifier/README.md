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

### Model Predictions and Probabilities
As in the [logistic regression](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/Logistic_Regression), 
we can build a model to make a prediction with real numbers, then using some function to transform these numbers to probabilities. 
In a multi-class problem, softmax function is used for this purpose. 
Considering the $$K$$ outputs are $$\\{ F_1(X),F_2(X),...,F_{K}(X) \\} = \\{ F_k(X) \\}^K_1$$
and the corresponding probability mass function are $$\\{ p_1(X),p_2(X),...,p_{K}(X) \\} = \\{ p_k(X) \\}^K_1$$, 
the softmax function gives:
```math
p_k(X) = \frac{e^{F_k(X)}}{\sum^K_{l=1}e^{F_l(X)}}
```
In this formula, $$X=(x_1, x_2, ... , x_N)$$ represents the input vector variable.

Since we are going to classify $$K$$ classes, we need $$K$$ different models, one for each class. 
It is convenient to one hot encode the target column in the dataset. 
Assuming our dataset contains a target column with $$n$$ data samples, after one hot encoding, 
we convert that column ($$n\times 1$$ array) to $$K$$ columns ($$n\times K$$ array). 
So we can build $$K$$ models to predict these $$K$$ columns.

As a first step, we can set all probabilities to $$p_k(X)=1/K$$, corresponding to $$F_k(X)=0$$.

### Pseudo Residuals
As other supervised learning method, we have to define a loss function and minimize it. 
For the multi-class problem, the loss function is 
```math
L(\{y_k, p_k(X)\}^K_1) = -\sum^K_{k=1} y_k\log{p_k(X)}
```
We can also rewrite this by the $$F_k(X)$$:
```math
L(\{y_k, F_k(X)\}^K_1) = -\sum^K_{k=1} y_k\log{\frac{e^{F_k(X)}}{\sum^K_{l=1}e^{F_l(X)}}}
```
With this formula, we can compute the derivative of this loss function:
```math
r_{ik} = -\frac{\partial L(\{y_{ik}, F_k(X_i)\}^K_1)}{\partial F_k(X_i)} = y_{ik} - p_k(X_i),
```
where $$i$$ represents the $$i$$-th samples and the negative sign is added for convenient. 
The $$r_{ik}$$ is so-called pseudo residual, which is just the negative gradient of the loss function. 

### From Raw Predictions to Probabilities
Now we can train $$K$$ models to fit the pseudo residuals. 
Since it becomes a problem to predict continuous variables, we should use a regression model like 
[decision tree regressor](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/Decision_Tree_Regressor).
However, this model can only predict the **pseudo** residuals rather than the $$F_k(X)$$.


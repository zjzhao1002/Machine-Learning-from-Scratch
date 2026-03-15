# XGBoost from Scratch

## Motivation
XGBoost (eXtreme Gradient Boosting) is a highly popular and 
effective algorithm that is used for both classification and regression tasks. 
It is a powerful implementation of gradient-boosted decision trees designed for speed and performance.
This algorithm is developed from 
[gradient boosting](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/Gradient_Boosting_Regressor) algorithm. 
To understand the maths and concepts, I build this model from scratch by **numpy** only. 

This [video](https://youtu.be/ZVFeW798-2I?si=bXlFUdfapr7Hwx_v) gives the mathematical details of XGBoost, 
and this [notebook](https://www.kaggle.com/code/josephhowerton/titanic-building-xgboost-from-scratch) gives the sample codes.
Of course, the [original paper](https://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf) 
and the [tutorial from XGBoost package](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) are also recommended. 

## Algorithm
For a given dataset with $$N$$ data samples and $$M$$ features, 
we use $$x_i$$ to represent the features of the $$i$$-th data sample, and $$y_i$$ is the corresponding variable.
Just like the normal gradient boosting, XGBoost combines the predictions of many "weak" models (typically simple decision trees) 
to create a single "strong" model. 
Assuming our model has $$K$$ trees, the prediction of the $$i$$-th data sample is 
```math
p_i = \phi(x_i) = f_0(x_i) + \eta\sum^K_{k=1} f_k(x_i),
```
where $$f_0(x_i)$$ is the initial prediction and $$f_k(x_i)$$ is the $$k$$-th weak model (tree). 
Here, the learning rate $$\eta$$ is introduced to prevent overfitting. 

### Objective Function
To train the XGBoost model, we have to minimize the following objective function:
```math
O(\phi) = \sum^N_{i=1} L(y_i, p_i) + \sum^K_{k=1}\Omega(f_k),
```
where $$L$$ is the loss function, and the $$\Omega$$ is the regularization term. 
For different problems, we have different loss functions. 
We use Mean Squared Error (MSE) for regression, and Binary Cross-Entropy (BCE) for binary classification.
The regularization term ($$\Omega$$) controls the complexity of the model to prevent overfitting. 
It penalizes trees that are too deep or have too many leaves. 
In the original paper of XGBoost, it is written as
```math
\Omega(f_k) = \gamma T + \frac{1}{2}\lambda\sum^T_{j=1}\omega^2_j,
```
where $$T$$ is the total number of leaves in this tree, and $$\omega_j$$ is the output value (weight) of the $$j$$-th leaf. 
$$\gamma$$ and $$\lambda$$ are two regularization parameters.
$$\gamma$$ is the penalty per leaf, which is also the minimum gain to split a node.
$$\lambda$$ penalizes leaf weights, which is also called Ridge or L2 regularization. 
It discourages the model from giving any single leaf too much "voting power." 

### Additive Training
To minimize the objective function, let us consider the $$t$$-th step of training. 
We should have:
```math
\begin{aligned}
  O^{(t)} &=& \sum^N_{i=1} L(y_i, p^{(t)}_i) + \sum^t_{k=1}\Omega(f_k) \\
          &=& \sum^N_{i=1} L(y_i, p^{(t-1)}_i + f_{t}(x_i)) + \Omega(f_t) + \text{constant}
\end{aligned}
```
For the loss function, we can do Taylor expansion:
```math
L(y_i, p^{(t-1)}_i + f_{t}(x_i)) = L(y_k, p^{(t-1)}_i) + \frac{\partial L}{\partial p^{(t-1)}_i}f_{t}(x_i) + \frac{1}{2}\frac{\partial^2 L}{\partial p^{(t-1)}_i}f^2_{t}(x_i) +...
```
The first-order derivative is called gradient, and the second-order derivative is called hessian. 
We can define:
```math
\begin{aligned}
  g_i &=& \frac{\partial L}{\partial p^{(t-1)}_i} \\
  h_i &=& \frac{\partial^2 L}{\partial p^{(t-1)}_i}
\end{aligned}
```
The objective function can be written to
```math
O^{(t)} \approx \sum^N_{i=1} \left[ L(y_i, p^{(t-1)}_i) + g_if_{t}(x_i) + \frac{1}{2}h_if^2_{t}(x_i) \right] + \gamma T + \frac{1}{2}\lambda\sum^T_{j=1}\omega^2_j + \text{constant}.
```
If $$x_i$$ is mapped to the $$j$$-th leaf, we have $$f_{t}(x_i) = \omega_j$$, and the previous equation becomes
```math
O^{(t)} \approx \sum^N_{i=1}L(y_i, p^{(t-1)}_i) + \sum^T_{j=1}\left[ (\sum_{i\in I_j}g_i)\omega_j + \frac{1}{2}(\sum_{i\in I_j}h_i+\lambda)\omega^2_{j} \right] + \gamma T + \text{constant}, 
```
where $$I_j$$ is the set of indices of data points assigned to the $$j$$-th leaf. 

Finally, we can miminize the objective function by calculating its derivative respect to $$\omega_j$$, and solve the following equation:
```math
\frac{\partial O^{(t)}}{\partial \omega_{j}} = \left[ \sum_{i\in I_j}g_i + (\sum_{i\in I_j}h_i+\lambda)\omega_{j} \right] = 0.
```
We get
```math
\omega_j = -\frac{\sum_{i\in I_j}g_i}{\sum_{i\in I_j}h_i+\lambda}.
```
We can insert this $$\omega_j$$ to the objective function and get:
```math
\tilde{O}^{(t)} = -\frac{1}{2}\sum^T_{j=1}\frac{(\sum_{i\in I_j}g_i)^2}{\sum_{i\in I_j}h_i+\lambda} + \gamma T.
```
Here I remove the loss function of previous step and constant term. This equation shows how good by adding the $$t$$-th tree.

### Learning the Tree Structure
Now we know how good a tree is, ideally we would enumerate all possible trees and pick the best one. 
In practice this is intractable, so we will try to optimize one level of the tree at a time. 
When we try to split a node into two leaves, and the score it gains is 
```math
Gain = \frac{1}{2}\left[ \frac{(\sum_{i\in I_L}g_i)^2}{\sum_{i\in I_L}h_i+\lambda} + \frac{(\sum_{i\in I_R}g_i)^2}{\sum_{i\in I_R}h_i+\lambda}
-\frac{(\sum_{i\in I}g_i)^2}{\sum_{i\in I}h_i+\lambda}  \right] - \gamma, 
```
where $$I=I_L\cup I_R$$. This just means we have a large gain if we get a smaller $$\tilde{O}$$ after splitting a node. 
We can see that $$\gamma$$ is the minimum gain, because the penalty of splitting one node into two leaves is exactly one $$\gamma$$.

### Regression
Previous equations can be applied to both regression and classification, but we have different loss functions for different problems.
For the regression problem, the loss function is Mean Squared Error (MSE): 
```math
\sum^N_{i=1} L(y_i, p_i) = \frac{1}{2}\sum^N_{i=1}(y_i-p_i)^2, 
```
where the $$1/2$$ is just added for convenient.
Note that the $$1/N$$ factor may be ignored when the dataset is large. The gradient and hessian are 
```math
\begin{aligned}
  g_i &=& \frac{\partial L}{\partial p_i} = p_i - y_i \\
  h_i &=& \frac{\partial^2 L}{\partial p_i} = 1
\end{aligned}
```
Now we can implement these equations to train the **XGBoostRegressor**.

### Classification
For the binary classification problem, the loss function is the Binary Cross-Entropy (BCE):
```math
\sum^N_{i=1} L(y_i, p_i) = \sum^N_{i=1}\left[ y_i\log{p_i} + (1-y_i)\log{(1-p_i)} \right]
```
Just like the [Gradient Boosting Classifier](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/Gradient_Boosting_Classifier), 
the predictions of the model are not exactly the probabilities of the class. 
Assuming the raw predictions are $$\hat{y}_i$$, the corresponding probabilities are given by the sigmoid function:
```math
p_i = \frac{1}{1+e^{-\hat{y}}}
```
So the gradient and hessian are 
```math
\begin{aligned}
  g_i &=& \frac{\partial L}{\partial \hat{y}_i} = p_i - y_i \\
  h_i &=& \frac{\partial^2 L}{\partial \hat{y}_i} = p_i(1-p_i)
\end{aligned}
```
These equations are used to train the **XGBoostClassifier**.

### Summary of Training 
I summarize the training steps in this subsection:
1. Initialize the model by constant value (mean value of $$y$$ for regression, 0.5 for classification).
2. Calculate the gradients and hessians.
3. Build a weak model (tree) by the gain formula.
4. Calculate the leaf values (weights).
5. Update the model predictions.
6. Repeat 2 through 5 for a specified number of iterations (`n_estimators`).

## Results
### Regression
I test the **XGBoostRegressor** by the [Student Performance](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression/) dataset. 
The hyperparameters are set to `n_estimators=6`, `learning_rate=0.25` and `max_depth=5` (the maximum depth of a tree).
The following table compares the first 10 values of test dataset (`y_true`) and model predictions (`y_pred`).

| y_true   | y_pred  |
| -------- | ------- |
| 84       | 80      |
| 67       | 64      |
| 72       | 70      |
| 17       | 24      |
| 71       | 66      |
| 68       | 69      |
| 39       | 44      |
| 36       | 38      |
| 47       | 49      |
| 39       | 43      |

The corresponding MSE are 
```math
\begin{aligned}
MSE_\text{train} = 18.432929787380942 \\
MSE_\text{test} = 19.28465183495664
\end{aligned}
```
It seems that this model works well.

### Classification
I test the **XGBoostClassifier** by the [500 Person Gender-Height-Weight-Body Mass Index](https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex) dataset. 
People with index 4 or 5 are obese.
The hyperparameters are set to `n_estimators=3`, `learning_rate=0.01` and `max_depth=4`. 
The accuracy can reach 97.3% for the training dataset, and 98% for the test dataset. 
The performance is very well.

# Conclusion
I build the XGBoost model from scratch by **numpy** only. 
It can be used for both regression and classification problems. 
The performance of this model is good. 
However, I just implement the fundamental parts of this algorithm. 
Some features such as **Cache-Awareness**, **Sparsity-Aware Splitting**, and **Weighted Quantile Sketch** are not implemented. 
These features actually make this model "Extreme".

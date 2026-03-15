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

To train the XGBoost model, we have to minimize the following objective function:
```math
O(\phi) = \sum^N_{i=1} L(y_i, p_i) + \sum^K_{k=1}\Omega(f_k),
```
where $$L$$ is the loss function, and the $$\Omega$$ is the regularization term. 
For different problems, we have different loss functions. 
We use Mean Square Error (MSE) for regression, and Binary Cross-Entropy (BCE) for binary classification.
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

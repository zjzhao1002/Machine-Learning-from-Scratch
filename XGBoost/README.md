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
\hat{y}_i = \phi(x_i) = f_0(x_i) + \eta\sum^K_{k=1} f_k(x_i),
```
where $$f_0(x_i)$$ is the initial prediction and $$f_k(x_i)$$ is the $$k$$-th weak model (tree). 
Here, the learning rate $$\eta$$ is introduced to prevent overfitting. 

To train the XGBoost model, we have to minimize the following objective function:
```math
O(\phi) = L(\phi) + \Omega(\phi)
```

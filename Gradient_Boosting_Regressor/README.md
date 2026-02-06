# Gradient Boosting Regressor from Scratch

## Motivation 
Gradient boosting is a machine learning technique used for regression and classification.
As a boosting technique, it gives a prediction model in the form of an ensemble of a sequence of weak models, 
which are typically simple decision trees.
When a decision tree is the weak learner, the resulting algorithm is called gradient-boosted trees.

I learned the maths and concepts by this [article](https://randomrealizations.com/posts/gradient-boosting-machine-from-scratch/) 
and this [notebook](https://www.kaggle.com/code/egazakharenko/gradient-boosting-from-scratch-full-tutorial).
This model is test by the [Student Performance](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression/) dataset. 

## Algorithm
Like other boosting methods, gradient boosting combines weak models into a single strong model iteratively. 
We are going to predict contineous variable $y$ by a vector of variables $x=(\text{Feature}_1, \text{Feature}_2, ..., \text{Feature}_N)$, so we can define a model
```math
\hat{y}_i = F(x_i), 
```
where $$\hat{y}_i$$ is the predicted values of this model. 
As other machine learning models, to train the model $$F$$, we have to minimize a lost function. 
For a regression model, a useful option is the mean squared error (MSE): 
```math
L_\text{MSE} = \frac{1}{n}\sum_{i}(y_i-F(x_i))^2
```

As a first step, we can use the mean value of $$y$$ of the training dataset as the initial prediction:
```math
F_0(x_i) = \bar{y}
```
This gives a residual:
```math
h_1(x_i) = y_i - F_0(x_i)
```
In the language of gradient boosting, we add a new weak model to fit $$h_1(x_i)$$, and give a new prediction:
```math
F_1(x_i) = F_0(x_i) + h_1(x_i)
```
After $$m$$ steps, the prediction is 
```math
F_m(x_i) = F_{m-1}(x_i) + h_m(x_i) = F_0(x_i) + \sum^m_{k=1} h_k(x_i),
```
and we have
```math
\frac{\partial L_\text{MSE}}{\partial F_m(x_i)} = \frac{2}{n}(y_i-F_m(x_i)) = \frac{2}{n}h_m(x_i)
```
This algorithm will minimize the residual during training, which also means minimizing the MSE.
The total steps for training is set by `n_estimators`.

If we just add weak model $$h_m(x_i)$$ to $$F_{m-1}(x_i)$$, we will get a perfect model for the training data, 
but it also means overfitting. 
To prevent overfitting, we introduce a parameter $$\eta$$ called **learning rate**:
```math
F_m(x_i) = F_{m-1}(x_i) + \eta h_m(x_i) = F_0(x_i) + \eta\sum^m_{k=1} h_k(x_i).
```
This is our final model. 

## Result
I use the [decision tree](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/Decision_Tree_Regressor) as the weak model. 
The hyperparameters are set to `n_estimators=6` and `max_depth=5`. 
The following table compares the first 10 values of test dataset (`y_true`) and model predictions (`y_pred`).

| y_true   | y_pred  |
| -------- | ------- |
| 84       | 81      |
| 67       | 63      |
| 72       | 71      |
| 17       | 21      |
| 71       | 67      |
| 68       | 70      |
| 39       | 44      |
| 36       | 37      |
| 47       | 48      |
| 39       | 42      |

The corresponding MSE are 
```math
\begin{aligned}
MSE_\text{train} = 11.136037767104325 \\
MSE_\text{test} = 11.756627099539498
\end{aligned}
```

It seems that this model works well, but is slightly overfitting. 

## Conclusion
I built a gradient boosting regressor from scratch to understand this algorithm. 
This model is used to predict student performance by the [Student Performance](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression/) dataset. 
The performance is good. 

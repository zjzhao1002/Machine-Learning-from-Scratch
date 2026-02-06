# Decision Tree Regressor from Scratch

## Motivation
Decision tree is a supervised learning approach algorithm, used for both classification and regression tasks. 
It is a simple model and easy to interpret. 

I have already built a [decision tree classifier](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/Decision_Tree_Classifier) to understand the key concepts.
In this mini project, I modify the previous codes to perform a regression. 
This model is test by the [Student Performance](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression/) dataset. 
The goal is predicting the student performance by using other features.

## Algorithm
### Lost Function
The main difference between a classifier and regressor is the lost function. 
In a regression problem, we are going to predict a continuous value, so we can just use variance as the lost function:
```math
S = \frac{1}{n}\sum_i(y_i-\bar{y})^2
```

### Information Gain
Information gain is a metric that indicates the improvement when making different partitions. 
Similar to the classifier, it is defined by
```math
IG = S_\text{node} - w_\text{left}S_\text{left}-w_\text{right}S_\text{right}
```
where
* $S_\text{node}$ is the variance of the current node.
* $S_\text{left}$ and $S_\text{right}$ are the variances at the left and right subsets after splitting.
* $w_\text{left}$ and $w_\text{right}$ are the proportion of data samples at the left and right subset, respectively.

### Predictions
The final difference is the prediction. 
All data points will go through the tree and reach the node they belonged to.
At the leaf nodes, we calculate their mean values. 
These leaf values are predictions. 

## Result
The following table compares the first 10 values of test dataset (`y_true`) and model predictions (`y_pred`).

| y_true   | y_pred |
| -------- | ------- |
| 84       | 85      |
| 67       | 63      |
| 72       | 76      |
| 17       | 18      |
| 71       | 63      |
| 68       | 71      |
| 39       | 42      |
| 36       | 32      |
| 47       | 46      |
| 39       | 42      |

As a toy model, this result is acceptable. To make better predictions, we can increase the `max_depth`. 
Normalizing the data should be helpful, too.

The performance of model can be assessed by calculating the Mean Square Error (MSE), which is defined by
```math
MSE = \frac{1}{n}\sum_i(y_\text{true}-y_\text{pred})^2
```
With the default setup, we have
```math
\begin{aligned}
MSE_\text{train} = 725.6361655218327 \\
MSE_\text{test} = 716.3561026814212
\end{aligned}
```
These results have been cross checked with the `DecisionTreeRegressor` from **sklearn**. 
The large values of MSE actually show the bad performance of this model. 
A normalization should be helpful to improve the performance.

## Conclusion
I built a decision tree regressor from scratch to understand this algorithm. 
This regressor is used to predict student performance by the [Student Performance](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression/) dataset. 
The performance of this regressor is not so good. 
It is still needed to be improved.

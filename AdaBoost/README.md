# AdaBoost from Scratch
## Introduction
Boosting is an ensemble learning method where we sequentially combines multiple weak models (or learners) to create a strong model.
AdaBoost is short for **Ada**ptive **Boost**ing. 
In this model, all data points have their weights. 
These weights are adaptive in the training process. 
After each iteration more weights are given to the misclassified points in the previous step. 
This process repeats and in the end all models are combined to make final predictions.

## Algorithm
### Decision Tree with Weighted Data
The first step of this algorithm is create a weak model which can handle weighted data. 
I develop the previous [decision tree](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/Decision_Tree) code for this purpose. 
With the weighted data, the probability of the certain class $x$ is 
```math
p(x) = \frac{\sum w(x)}{W}
```
where $w(x)$ is a weight with class $x$ and $W$ is the sum of all weights in dataset.

This probability is used to calculate the entropy:
```math
H(x) = -\sum_{x\in \chi}p(x)\log_2p(x)
```
In this case, we have to modify the formula for information gain:
```math
IG = \frac{W_t}{W} \left(H_\text{node} - \frac{W_L}{W_t}H_\text{left}-\frac{W_R}{W_t}H_\text{right}\right)
```
where $W_t$ is the weighted sum of the current node, $W_L$ is the weighted sum of the left child, 
and $W_R$ is the weighted sum of the right node, respectively.

The decision tree is built by this new setup.

### Weighted Error and Importance
Assuming we have a AdaBoost model with $n$ weak models (decision trees). 
In the $m$-th iteration, the weighted error is defined by 
```math
\epsilon_{m} = \frac{\sum_{y_{\text{true}}\neq y_{\text{pred}}}w^{(m)}_i}{\sum^N_{i=1}w^{(m)}_i}
```
where $y_{\text{true}}$ and $y_{\text{pred}}$ are the true and predicted target values, respectively. 
$N$ is the total number of data samples. 

With this weighted error, we can calculate a value called importance (amount of say or influence):
```math
\alpha_{m} = \frac{1}{2}\log\left( \frac{1-\epsilon_m}{\epsilon_m} \right)
```
The derivation of the formula can be found on [Wikipedia](https://en.wikipedia.org/wiki/AdaBoost).
The $\alpha_{m}$ indicates the importance of the $m$-th iteration. 

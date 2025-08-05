# AdaBoost from Scratch
## Introduction
Boosting is an ensemble learning method where we sequentially combines multiple weak models (or learners) to create a strong model.
AdaBoost is short for **Ada**ptive **Boost**ing. 
In this model, all data points have their weights. 
These weights are adaptive in the training process. 
After each iteration more weights are given to the misclassified points in the previous step. 
This process repeats and in the end all models are combined to make final predictions. 
I learned the concepts and maths by this [blog](https://blog.devgenius.io/adaboost-from-scratch-f8979d961948), 
this [notebook](https://www.kaggle.com/code/egazakharenko/adaboost-samme-r2-from-scratch-using-python) and 
[Wikipedia](https://en.wikipedia.org/wiki/AdaBoost).

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

### Update The Weights
After finishing $m$-th iteration, the weight should be updated by
```math
w_i^{(m+1)} = w_i^{(m)}\exp{\alpha_m}(y_{\text{true}}\neq y_{\text{pred}})
```
This formula give more weights to the misclassified points. 
Finally, the weights should be normilized:
```math
w_i^{(m+1)} \to \frac{w_i^{(m+1)}}{\sum{w_i^{(m+1)}}}
```

### Train The AdaBoost Model
With the concepts and formulae above, we can train the AdaBoost model by following steps:
1. Initialize the sample weights by $1/N$ for data samples.
2. A decision tree is trained by using these weights.
3. Calculate the weighted error and $\alpha$.
4. Update the sample weights by the $\alpha$ value.
5. Repeat step 2~4 `n_estimators` times.

### Prediction
Each weak models should have their predictions $y^{(m)}_{\text{pred}}$. The final predictions of the AdaBoost model is
```math
y_{\text{final}} = \sum_{m} \alpha_m y^{(m)}_{\text{pred}}
```
As the decision tree project, this classifier is test by predicting if a person is obese by using this 
[500 Person Gender-Height-Weight-Body Mass Index](https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex) 
With the setup in the main.py file, I got the 84% accuracy for the training data, while the accuracy for the test data is 85%.

## Conclusion
AdaBoost is an ensemble method with sequential decision trees. These trees must be trained by weighted samples. 
The sample weights should be updated `n_estimators` times in the training process. 

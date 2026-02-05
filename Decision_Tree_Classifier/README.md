# Decision Tree Classifier from Scratch

## Motivation
Decision tree is a supervised learning approach algorithm, used for both classification and regression tasks. 
It is a simple model and easy to interpret. 

To understand the key concepts and maths of decision tree, I built a decision tree classifier from scratch by using **numpy** only.
The purpose of this classifier is to predict if a person is obese by using this 
[500 Person Gender-Height-Weight-Body Mass Index](https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex) dataset. 
I mainly follow this [blog](https://anderfernandez.com/en/blog/code-decision-tree-python-from-scratch/) and 
this [notebook](https://www.kaggle.com/code/fareselmenshawii/decision-tree-from-scratch/).

## Algorithm
A typical decision tree is looked like:
![decision tree](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/blob/main/Decision_Tree_Classifier/decision_tree.png)

A tree is built by splitting the source dataset, constituting the root node of the tree, 
into subsets, which constitute the successor children. These subsets are also called sub-tree.
The splitting is based on a set of splitting rules based on classification features. 
This process is repeated on each derived subset in a recursive manner called recursive partitioning. 
The recursion is completed when the subset at a node has all the same values of the target variable, 
or when splitting no longer adds value to the predictions.
This last node is known as a leaf node. 

### Impurity and Lost Function
As in other machine learning models, the lost function is the basis of this algorithm. 
The most useful lost function is **entropy** in decision tree.

Entropy is defined by
```math
H(x) = -\sum_{x\in \chi}p(x)\log_2p(x)
```
where $p(x)$ is the probability of the certain class $x$. 
$\chi$ is the set of all classes, and $\sum$ is the sum over the possible classes.

Entropy is a way to measure impurity or randomness in data points. 
It goes from 0 to 1. High entropy means these data points are impure and they are belonged to different classes. 
In contrast, we get low entropy if the splitted data group have a dominant class.
So we want to search for splits with the lowest entropy.

### Information Gain
Information gain is a metric that indicates the improvement when making different partitions. 
It is defined by
```math
IG = H_\text{node} - w_\text{left}H_\text{left}-w_\text{right}H_\text{right}
```
where
* $H_\text{node}$ is the entropy of the current node.
* $H_\text{left}$ and $H_\text{right}$ are the entropies at the left and right subsets after splitting.
* $w_\text{left}$ and $w_\text{right}$ are the proportion of data samples at the left and right subset, respectively.

The best split should have the highest information gain. 
We can loop over all features in the dataset and calculate the information gain, 
and choose the split with highest information gain to split data. 

### Build The Decision Tree
To build a decision tree, we have to define 3 hyperparemeters: 
* `max_depth`: The maximum depth of the decision tree.
* `min_samples`: The minimum number of data samples required to split an internal node.
* `min_information_gain`: The minimum information gain required to split an internal node.

With these hyperparemeters, we can build a decision tree step by step:
1. Make sure the conditions established by `max_depth` and `min_simple` are being fulfilled.
2. Calculate the information gain for all features.
3. Choose the split with the highest information gain.
4. Make sure the highest information gain is greater than the `min_information_gain`.
5. Split dataset by the best split into left and right dataset.
6. Repeat 1~5 by using the children dataset until one of the conditions in step 1 or 4 is not fulfilled.
7. The node that does not fulfilled the conditions above is a leaf node.
We get the most occuring classes in the dataset of this node as the prediction.

### Make The Predictions
After building the decision tree, we can use it to make the predictions. 
We can input the feature matrix to get the predictions. 
All data points will go through the tree and reach the node they belonged to.
The leaf values are their predictions. 
With the default setup in my code, the accuracy can reach 86% for the training dataset, and 80% for the test dataset.

## Conclusion
Decision tree is a simple machine learning algorithm. 
I built it from scratch to classify the [500 Person Gender-Height-Weight-Body Mass Index](https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex) dataset. 
In principle, decision tree can be also used for regression problem. 
We just need to make small modification to this code. I may do that in the future.

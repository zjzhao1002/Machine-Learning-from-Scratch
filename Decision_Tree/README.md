# Decision Tree from Scratch

## Motivation
Decision tree is a supervised learning approach algorithm, used for both classification and regression tasks. 
It is a simple model and easy to interpret. 

To understand the key concepts and maths of decision tree, I built a decision tree classifier from scratch by using **numpy** only.
The purpose of this classifier is to predict if a person is obese by using this 
[500 Person Gender-Height-Weight-Body Mass Index](https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex) dataset. 
I mainly follow this [blog](https://anderfernandez.com/en/blog/code-decision-tree-python-from-scratch/) and 
this [notebook](https://www.kaggle.com/code/fareselmenshawii/decision-tree-from-scratch/).

## Maths
A typical decision tree is looked like:
![decision tree](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/blob/main/Decision_Tree/decision_tree.png)

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

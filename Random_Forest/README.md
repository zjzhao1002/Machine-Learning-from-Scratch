# Random Forest from Scratch

## Motivation
Random forest or random decision forest is an ensemble learning method for classification, 
regression and other tasks that works by creating a multitude of decision trees during training. 
For classification tasks, the output of the random forest is the class selected by most trees. 
For regression tasks, the output is the average of the predictions of the trees. 

To understand the key concepts and algorithm of random forest, I built a random forest classifier from scratch. 
This is based on the [decision tree I built](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/Decision_Tree). 
As the decision tree project, this classifier is test by predicting if a person is obese by using this 
[500 Person Gender-Height-Weight-Body Mass Index](https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex) dataset. 
The main references are this [article](https://carbonati.github.io/posts/random-forests-from-scratch/) 
and this [notebook](https://www.kaggle.com/code/fareselmenshawii/random-forest-from-scratch). 

## Algorithm
As mentioned above, random forest is a collection of decision trees. 
A single tree need to grow very deep to make strong predictions. 
However, decision trees have low bias and high variance. 
This just means that it is inconsistent, but accurate on average.
Random forests are a way of averaging multiple deep decision trees, 
trained on different parts of the same training set, with the goal of reducing the variance. 
To do that, a method called bagging (**b**ootstrap **agg**regat**ing**) is used. 

### Bagging
Given a standard dataset with size `n_samples`, bagging generates `num_trees` new datasets. 
`num_trees` is also the number of trees. Each dataset has `bootstrap_samples` size. 
By sampling original dataset with replacement, some observations may be repeated in each bootsrap dataset. 
Sampling with replacement ensures each bootstrap is independent from its peers, 
as it does not depend on previous chosen samples when sampling. 
Then, `num_trees` decision trees are fitted using the above bootstrap samples 
and combined by voting for classification. 

When there are too many features in the dataset, we can also bootsrap dataset by `max_features`. 
There are two useful options for `max_features`: `sqrt` and `log`. 
If the total features in the original is $n_f$, the $\sqrt{n_f}$ and $\log_2(n_f)$ are chosen, respectively. 
For our small dataset, it is not necessary to use these options. 

### Make the predictions
The predicitons of the random forest is the most selected classes of all trees. 
With the setup in the `main.py` file, I got the 94% accuracy for the training data, 
while the accuracy for the test data is 92%.

## Conclusion
Random forest is a model with many decision trees, and its predictions are averaged of these trees. 
It is easy to understand. The only trick is bagging in this case.

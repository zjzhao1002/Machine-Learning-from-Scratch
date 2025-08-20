# Convolutional Neural Network (CNN) from Scratch

## Introduction

A convolutional neural network (CNN) is a particular type of neural network for computer vision tasks. 
It is a powerful tool to process image data. Image data are high-dimensional. 
A typical image for a classification task contains $224\times 224$ RGB values. 
If we use a fully connected neural network with linear layers, we have a huge number of parameters to optimized. 
In addition, nearby image pixels are related. Fully connected neural network cannot treat this relationship. 
Finally, a geometric transformation of image should not change the interpretation of an image. 
A fully connected model must learn all patterns of pixels, which is inefficient. 

A convolutional layer can process each local image region independently, 
using parameters (weights and biases) shared across the whole image. 
It has fewer parameters than the fully connected layer, can understand the spatial relationship, 
and do not need to re-learn the interpretation of the pixels at every position. 
A CNN is a network that predominantly consists convolutional layers. 

In this project, I build a CNN by using **numpy** only. 
The goal of this network is classified the [Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) dataset. 
I learned CNN by this [book](https://udlbook.github.io/udlbook/) and this [article](https://www.quarkml.com/2023/07/build-a-cnn-from-scratch-using-python.html).

# Machine Learning from Scratch

Building fundamental machine learning algorithms from the ground up using **Python** and **NumPy**.

## Project Overview

This repository is an educational journey into the inner workings of machine learning. By implementing these algorithms from scratch, I aim to move beyond "black-box" libraries like Scikit-Learn or TensorFlow and gain a deep, mathematical understanding of how these models truly function.

The focus is on clarity, mathematical rigor, and efficient implementation using vectorized operations in NumPy.

---

## Algorithms Implemented

### Supervised Learning: Regression
*   **[Linear Regression](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/Linear_Regression)**: Simple linear modeling using gradient descent.
*   **[Decision Tree (Regression)](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/Decision_Tree_Regressor)**: Tree-based regression splitting on MSE reduction.
*   **[Gradient Boosting Machine (Regression)](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/Gradient_Boosting_Regressor)**: Ensemble technique using additive training for regression.

### Supervised Learning: Classification
*   **[Logistic Regression](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/Logistic_Regression)**: Binary classification using the sigmoid function and cross-entropy loss.
*   **[Decision Tree (Classification)](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/Decision_Tree_Classifier)**: Classification tree using Gini impurity or entropy for splits.
*   **[Random Forest](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/Random_Forest)**: Ensemble method using bagging and random feature selection.
*   **[Adaptive Boosting (AdaBoost)](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/AdaBoost)**: Boosting method focusing on misclassified samples.
*   **[Gradient Boosting Machine (Classification)](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/Gradient_Boosting_Classifier)**: GBM implementation adapted for binary classification.
*   **[XGBoost](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/XGBoost)**: eXtreme Gradient Boosting with regularization and Taylor expansion of the loss function.

### Deep Learning
*   **[(Fully Connected) Neural Network](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/Neural_Network)**: Multi-layer perceptron with backpropagation and ReLU/Softmax activations. Tested on MNIST.
*   **[Convolutional Neural Network](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/Convolutional_Neural_Network)**: CNN implementation featuring convolution, pooling, and flattening layers. Tested on Fashion-MNIST.

### Unsupervised Learning
*   **[K-Means Clustering](https://github.com/zjzhao1002/Machine-Learning-from-Scratch/tree/main/KMeans)**: Classic centroid-based clustering algorithm.

---

## Key Features

-   **NumPy-First:** All core logic is implemented using NumPy for efficient numerical and matrix operations.
-   **Consistent API:** Models generally follow a consistent `fit(X, y)` and `predict(X)` interface, making them easy to test and swap.
-   **Pure Implementation:** Avoids high-level machine learning frameworks for the model logic itself.
-   **Visualization:** Many implementations include loss curves or result visualizations using Matplotlib.

## Getting Started

### Prerequisites

-   Python 3.x
-   NumPy
-   Pandas (for data loading)
-   Matplotlib (for visualization)
-   Scikit-Learn (only for data preprocessing utilities like `train_test_split`)
-   idx2numpy (for MNIST-style datasets)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/zjzhao1002/Machine-Learning-from-Scratch.git
    cd Machine-Learning-from-Scratch
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running an Example

Each algorithm is isolated in its own directory with a `main.py` script demonstrating its usage.

```bash
cd XGBoost
python main_classification.py
```

---

## Project Structure

```text
Machine-Learning-from-Scratch/
├── AdaBoost/                      # AdaBoost implementation
├── Convolutional_Neural_Network/  # CNN from scratch
├── Decision_Tree_Classifier/      # Tree-based classification
├── KMeans/                        # K-Means clustering
├── Linear_Regression/             # Simple linear regression
├── Neural_Network/                # Fully connected ANN
└── ...                            # Other algorithms
```

Each directory typically contains:
-   `<Algorithm>.py`: The model implementation.
-   `main.py`: Demo script.
-   `README.md`: Detailed documentation and mathematical derivation for that specific algorithm.

---

## Future Roadmap / TODO

To further expand the scope of this project and deepen the understanding of machine learning, the following models and features are planned for implementation:

- [ ] **Support Vector Machines (SVM):** Implementing the dual optimization problem and kernel trick.
- [ ] **K-Nearest Neighbors (KNN):** A simple but powerful distance-based algorithm.
- [ ] **Naive Bayes:** Probabilistic classification based on Bayes' Theorem.
- [ ] **Principal Component Analysis (PCA):** Dimensionality reduction using Eigen-decomposition or SVD.
- [ ] **Recurrent Neural Networks (RNN/LSTM):** Handling sequential data and time-series analysis.
- [ ] **DBSCAN:** Density-based spatial clustering of applications with noise.
- [ ] **Optimization Algorithms:** Implementing advanced optimizers like Adam, RMSprop, and Adagrad for neural networks.

---

## Related Projects

-   **[Generative Pre-Trained Transformer (GPT) from Scratch](https://github.com/zjzhao1002/GPT-from-Scratch)**: An implementation of a GPT-style transformer model.

---

## References & Inspiration

-   [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen.
-   [Understanding Deep Learning](https://udlbook.github.io/udlbook/) by Simon J.D. Prince.
-   [XGBoost Official Documentation](https://xgboost.readthedocs.io/).

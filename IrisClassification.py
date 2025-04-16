"""
Perceptron and Naive Bayes Comparison on Iris Dataset

This script implements the third task from Practical 3:
- Load the iris dataset
- Implement a perceptron algorithm
- Train a Naive Bayes classifier
- Compare the accuracy of both models
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(3)

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

class Perceptron:
    """Simple perceptron implementation"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize the perceptron
        
        Parameters:
        -----------
        learning_rate : float, default=0.01
            Learning rate for weight updates
        n_iterations : int, default=1000
            Number of training iterations
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """
        Train the perceptron
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : object
            Fitted perceptron
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Convert y to binary for binary classification
        # For multiclass, we can use one-vs-rest approach
        y_ = np.where(y > 0, 1, 0)
        
        # Training loop
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = 1 if linear_output >= 0 else 0
                
                # Update weights and bias if prediction is wrong
                if y_[idx] != y_pred:
                    update = self.learning_rate * (y_[idx] - y_pred)
                    self.weights += update * x_i
                    self.bias += update
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples
        
        Returns:
        --------
        y : array-like of shape (n_samples,)
            Predicted class labels
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = np.where(linear_output >= 0, 1, 0)
        return y_pred

def main():
    """Main function for perceptron and Naive Bayes comparison task"""
    # Load iris dataset
    print("Loading iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature names: {feature_names}")
    print(f"Target names: {target_names}")
    print(f"Class distribution:\n{pd.Series(y).value_counts()}")
    
    # For binary classification with perceptron, we'll use only two classes
    # (e.g., setosa vs. non-setosa)
    # Alternatively, we could implement one-vs-rest approach for multiclass
    X_binary = X.copy()
    y_binary = (y == 0).astype(int)  # 1 for setosa, 0 for others
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_binary, y_binary, test_size=0.3, random_state=3
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate perceptron
    print("\nTraining perceptron...")
    perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)
    perceptron.fit(X_train_scaled, y_train)
    
    # Get perceptron predictions
    y_pred_perceptron = perceptron.predict(X_test_scaled)
    
    # Calculate perceptron accuracy
    accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)
    print(f"Perceptron accuracy: {accuracy_perceptron:.4f}")
    
    # Create confusion matrix for perceptron
    cm_perceptron = confusion_matrix(y_test, y_pred_perceptron)
    
    # Train and evaluate Naive Bayes
    print("\nTraining Naive Bayes classifier...")
    nb = GaussianNB()
    nb.fit(X_train, y_train)  # Naive Bayes doesn't require scaling
    
    # Get Naive Bayes predictions
    y_pred_nb = nb.predict(X_test)
    
    # Calculate Naive Bayes accuracy
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    print(f"Naive Bayes accuracy: {accuracy_nb:.4f}")
    
    # Create confusion matrix for Naive Bayes
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    
    # Compare accuracies
    print("\nModel comparison:")
    print(f"Perceptron accuracy: {accuracy_perceptron:.4f}")
    print(f"Naive Bayes accuracy: {accuracy_nb:.4f}")
    
    if accuracy_perceptron > accuracy_nb:
        print("Perceptron has better accuracy on the iris dataset")
    elif accuracy_nb > accuracy_perceptron:
        print("Naive Bayes has better accuracy on the iris dataset")
    else:
        print("Both models have the same accuracy on the iris dataset")
    
    # Plot confusion matrices
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_perceptron, annot=True, fmt='d', cmap='Blues')
    plt.title('Perceptron Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues')
    plt.title('Naive Bayes Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('output/iris_confusion_matrices.png')
    
    # Plot accuracies as bar chart
    plt.figure(figsize=(8, 6))
    models = ['Perceptron', 'Naive Bayes']
    accuracies = [accuracy_perceptron, accuracy_nb]
    
    plt.bar(models, accuracies)
    plt.ylim([0, 1])
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison on Iris Dataset')
    
    # Add accuracy values on top of bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    plt.savefig('output/iris_accuracy_comparison.png')
    
    # Generate a report
    report = """# Task 3: Perceptron and Naive Bayes Comparison on Iris Dataset

## Model Comparison Results

- Perceptron Accuracy: {:.4f}
- Naive Bayes Accuracy: {:.4f}

## Conclusion

{}

## Explanation

The iris dataset is a classic dataset in machine learning, containing measurements
of sepal length, sepal width, petal length, and petal width for three species of iris
flowers: setosa, versicolor, and virginica.

For this task, we simplified the problem to binary classification (setosa vs. non-setosa)
to match the perceptron implementation which is designed for binary classification.
However, Naive Bayes naturally handles multiclass problems.

The perceptron is a simple linear classifier that learns a decision boundary to separate
classes. It works well when classes are linearly separable. Naive Bayes, on the other hand,
is a probabilistic classifier based on applying Bayes' theorem with independence assumptions.

The difference in accuracy between these models on the iris dataset indicates which approach
is better suited for this particular data structure and classification task.
""".format(
        accuracy_perceptron,
        accuracy_nb,
        "Perceptron has better accuracy on the iris dataset" if accuracy_perceptron > accuracy_nb else
        "Naive Bayes has better accuracy on the iris dataset" if accuracy_nb > accuracy_perceptron else
        "Both models have the same accuracy on the iris dataset"
    )
    
    # Write report to file
    with open('output/iris_comparison_report.md', 'w') as f:
        f.write(report)
    
    print("\nTask completed. Generated files:")
    print("- output/iris_confusion_matrices.png")
    print("- output/iris_accuracy_comparison.png")
    print("- output/iris_comparison_report.md")

if __name__ == '__main__':
    main()
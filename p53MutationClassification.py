"""
P53 Mutation Classification with SVM

This script implements the second task from Practical 3:
- Load the p53 mutation dataset
- Perform 3-fold cross-validation for SVM classification
- Plot ROC curves and calculate AUC for each fold
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(3)

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

def load_p53_data(file_path):
    """Load the p53 mutation dataset"""
    print(f"Loading p53 mutation dataset from {file_path}...")
    
    try:
        # Load data from file
        p53_data = pd.read_csv(file_path, sep='\t')
        print(f"Dataset shape: {p53_data.shape}")
        
        # Check if the dataset has the expected structure
        print("Column names:", p53_data.columns.tolist())
        
        # Assuming the last column is the class label
        X = p53_data.iloc[:, :-1]
        y = p53_data.iloc[:, -1]
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Class distribution:\n{y.value_counts()}")
        
        return X, y
    
    except Exception as e:
        print(f"Error loading p53 dataset: {e}")
        raise

def main(data_file='Kato_p53_mutants_200.txt'):
    """Main function for p53 mutation classification task"""
    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found")
        print("Please make sure the data file is in the current directory")
        return
    
    # Load p53 mutation data
    X, y = load_p53_data(data_file)
    
    # Perform 3-fold cross-validation
    print("\nPerforming 3-fold cross-validation...")
    
    # Initialize arrays to store results
    tprs = []
    aucs = []
    accuracies = []
    mean_fpr = np.linspace(0, 1, 100)
    
    # Initialize plot for CV ROC curves
    plt.figure(figsize=(10, 8))
    
    # Initialize KFold
    kf = KFold(n_splits=3, shuffle=True, random_state=3)
    
    # Perform cross-validation
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train SVM with radial basis function kernel
        svm = SVC(kernel='rbf', probability=True, random_state=3)
        svm.fit(X_train, y_train)
        
        # Get predictions
        y_pred = svm.predict(X_test)
        y_pred_prob = svm.predict_proba(X_test)[:, 1]
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # Interpolate TPR at mean FPR points
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        # Plot ROC curve for this fold
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i+1} (AUC = {roc_auc:.4f})')
        
        print(f"Fold {i+1}: Accuracy = {accuracy:.4f}, AUC = {roc_auc:.4f}")
    
    # Calculate mean and std of TPRs across all folds
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    # Plot mean ROC curve
    plt.plot(mean_fpr, mean_tpr, 'b-', lw=2,
             label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for 3-fold Cross-Validation - P53 Mutation Classification')
    plt.legend(loc="lower right")
    plt.savefig('output/p53_roc_cv.png')
    
    # Plot boxplot of AUCs
    plt.figure(figsize=(8, 6))
    plt.boxplot(aucs)
    plt.ylabel('AUC')
    plt.title('Distribution of AUC values across 3 folds - P53 Mutation Classification')
    plt.savefig('output/p53_auc_boxplot.png')
    
    print(f"\nCross-validation results:")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    print("\nTask completed. Generated plots:")
    print("- output/p53_roc_cv.png")
    print("- output/p53_auc_boxplot.png")

if __name__ == '__main__':
    import sys
    
    # Use provided data file path if available
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
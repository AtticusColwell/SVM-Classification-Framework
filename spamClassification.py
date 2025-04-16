"""
Spam Classification with SVM

This script implements the first task from Practical 3:
- Load the spam dataset
- Split into 80% training / 20% test set
- Train an SVM on the training set
- Plot ROC curve on the test set
- Perform 10-fold cross-validation and plot ROC curves
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(3)

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

def load_spam_data():
    """Load the spam dataset"""
    print("Loading spam dataset...")
    
    try:
        # Try to load from scikit-learn
        spam_data = fetch_openml(name='spambase', version=1, as_frame=True)
        X = spam_data.data
        y = (spam_data.target == '1').astype(int)  # Convert to binary
    except Exception as e:
        print(f"Error loading spam dataset: {e}")
        print("Trying alternative method...")
        
        # Alternative method - directly from UCI
        spam_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
        column_names = [f"feat_{i}" for i in range(57)] + ["is_spam"]
        spam_data = pd.read_csv(spam_url, names=column_names)
        X = spam_data.iloc[:, :-1]
        y = spam_data.iloc[:, -1]
    
    print(f"Loaded spam dataset with {X.shape[0]} samples and {X.shape[1]} features")
    return X, y

def main():
    """Main function for spam classification task"""
    # Load spam data
    X, y = load_spam_data()
    
    # Split data into training and testing sets (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=3
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train SVM with radial basis function kernel
    print("Training SVM with RBF kernel...")
    svm = SVC(kernel='rbf', probability=True, random_state=3)
    svm.fit(X_train, y_train)
    
    # Evaluate on test set and plot ROC curve
    print("Evaluating on test set...")
    y_pred_prob = svm.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve for test set
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Spam Classification - Test Set')
    plt.legend(loc="lower right")
    plt.savefig('output/spam_roc_test.png')
    print(f"Test AUC: {roc_auc:.4f}")
    
    # Perform 10-fold cross-validation
    print("\nPerforming 10-fold cross-validation...")
    
    # Initialize arrays to store results
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    # Initialize plot for CV ROC curves
    plt.figure(figsize=(10, 8))
    
    # Initialize KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=3)
    
    # Perform cross-validation
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Split data
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
        
        # Standardize features
        scaler_cv = StandardScaler()
        X_train_cv = scaler_cv.fit_transform(X_train_cv)
        X_test_cv = scaler_cv.transform(X_test_cv)
        
        # Train SVM
        svm = SVC(kernel='rbf', probability=True, random_state=3)
        svm.fit(X_train_cv, y_train_cv)
        
        # Get predictions
        y_pred_prob_cv = svm.predict_proba(X_test_cv)[:, 1]
        
        # Calculate ROC curve and AUC
        fpr_cv, tpr_cv, _ = roc_curve(y_test_cv, y_pred_prob_cv)
        roc_auc_cv = auc(fpr_cv, tpr_cv)
        aucs.append(roc_auc_cv)
        
        # Interpolate TPR at mean FPR points
        interp_tpr = np.interp(mean_fpr, fpr_cv, tpr_cv)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        # Plot ROC curve for this fold
        plt.plot(fpr_cv, tpr_cv, alpha=0.3, lw=1, label=f'Fold {i+1} (AUC = {roc_auc_cv:.4f})')
        
        print(f"Fold {i+1}: AUC = {roc_auc_cv:.4f}")
    
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
    plt.title('ROC Curves for 10-fold Cross-Validation - Spam Classification')
    plt.legend(loc="lower right")
    plt.savefig('output/spam_roc_cv.png')
    
    # Plot boxplot of AUCs
    plt.figure(figsize=(8, 6))
    plt.boxplot(aucs)
    plt.ylabel('AUC')
    plt.title('Distribution of AUC values across 10 folds - Spam Classification')
    plt.savefig('output/spam_auc_boxplot.png')
    
    print(f"\nCross-validation results:")
    print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    
    print("\nTask completed. Generated plots:")
    print("- output/spam_roc_test.png")
    print("- output/spam_roc_cv.png")
    print("- output/spam_auc_boxplot.png")
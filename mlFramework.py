import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(3)

class MLFramework:
    """Main framework class for machine learning tasks"""
    
    def __init__(self, task_name, data_path=None, test_size=0.2, n_folds=10):
        """
        Initialize the ML framework
        
        Parameters:
        -----------
        task_name : str
            Name of the ML task ('spam', 'p53', 'iris', or custom)
        data_path : str, optional
            Path to the dataset file
        test_size : float, default=0.2
            Proportion of the dataset to include in the test split
        n_folds : int, default=10
            Number of folds for cross-validation
        """
        self.task_name = task_name
        self.data_path = data_path
        self.test_size = test_size
        self.n_folds = n_folds
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
    def load_data(self):
        """Load and preprocess the dataset based on the task"""
        print(f"Loading data for {self.task_name} task...")
        
        if self.task_name == 'spam':
            # For spam task, use kernlab's spam dataset
            try:
                from sklearn.datasets import fetch_openml
                spam_data = fetch_openml(name='spambase', version=1, as_frame=True)
                self.X = spam_data.data
                self.y = (spam_data.target == '1').astype(int)  # Convert to binary
                print(f"Loaded spam dataset with {self.X.shape[0]} samples and {self.X.shape[1]} features.")
            except Exception as e:
                print(f"Error loading spam dataset: {e}")
                print("Trying alternative method to load spam data...")
                # Alternative method to load spam data
                spam_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
                try:
                    column_names = [f"feat_{i}" for i in range(57)] + ["is_spam"]
                    spam_data = pd.read_csv(spam_url, names=column_names)
                    self.X = spam_data.iloc[:, :-1]
                    self.y = spam_data.iloc[:, -1]
                    print(f"Loaded spam dataset with {self.X.shape[0]} samples and {self.X.shape[1]} features.")
                except Exception as e:
                    print(f"Failed to load spam dataset: {e}")
                    raise
        
        elif self.task_name == 'p53':
            # For p53 task
            if self.data_path is None:
                raise ValueError("Data path must be provided for p53 task")
            
            try:
                p53_data = pd.read_csv(self.data_path, sep='\t')
                # Assuming the last column is the class label
                self.X = p53_data.iloc[:, :-1]
                self.y = p53_data.iloc[:, -1]
                print(f"Loaded p53 dataset with {self.X.shape[0]} samples and {self.X.shape[1]} features.")
            except Exception as e:
                print(f"Failed to load p53 dataset: {e}")
                raise
        
        elif self.task_name == 'iris':
            # For iris task
            from sklearn.datasets import load_iris
            iris = load_iris()
            self.X = pd.DataFrame(iris.data, columns=iris.feature_names)
            self.y = iris.target
            print(f"Loaded iris dataset with {self.X.shape[0]} samples and {self.X.shape[1]} features.")
        
        elif self.data_path is not None:
            # For custom datasets
            try:
                file_ext = os.path.splitext(self.data_path)[1].lower()
                if file_ext == '.csv':
                    data = pd.read_csv(self.data_path)
                elif file_ext == '.tsv' or file_ext == '.txt':
                    data = pd.read_csv(self.data_path, sep='\t')
                elif file_ext == '.xlsx' or file_ext == '.xls':
                    data = pd.read_excel(self.data_path)
                else:
                    raise ValueError(f"Unsupported file format: {file_ext}")
                
                # Assuming the last column is the class label
                self.X = data.iloc[:, :-1]
                self.y = data.iloc[:, -1]
                print(f"Loaded custom dataset with {self.X.shape[0]} samples and {self.X.shape[1]} features.")
            except Exception as e:
                print(f"Failed to load custom dataset: {e}")
                raise
        else:
            raise ValueError("Either task_name or data_path must be provided")
        
        # Perform data validation
        self._validate_data()
        return self
    
    def _validate_data(self):
        """Validate the loaded data"""
        if self.X is None or self.y is None:
            raise ValueError("Data not loaded properly")
        
        # Check for missing values
        if self.X.isnull().values.any():
            print("Warning: Dataset contains missing values")
            # Fill missing values with median
            self.X = self.X.fillna(self.X.median())
        
        # Check for non-numeric features
        if not np.issubdtype(self.X.dtypes.dtype, np.number):
            print("Warning: Dataset contains non-numeric features. Converting...")
            # Convert categorical features to one-hot encoding
            self.X = pd.get_dummies(self.X)
        
        # Standardize features
        scaler = StandardScaler()
        self.X = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns)
        
        # Convert y to appropriate format based on task
        if self.task_name in ['spam', 'p53']:
            # Binary classification
            self.y = self.y.astype(int)
        
        print("Data validation complete")
    
    def split_data(self):
        """Split the dataset into training and testing sets"""
        print("Splitting data into training and testing sets...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=3
        )
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Testing set: {self.X_test.shape[0]} samples")
        return self
    
    def train_svm(self, kernel='rbf', C=1.0):
        """Train a Support Vector Machine model"""
        print(f"Training SVM with {kernel} kernel...")
        self.model = SVC(kernel=kernel, C=C, probability=True, random_state=3)
        self.model.fit(self.X_train, self.y_train)
        print("SVM training complete")
        return self
    
    def train_naive_bayes(self):
        """Train a Naive Bayes model"""
        print("Training Naive Bayes model...")
        self.model = GaussianNB()
        self.model.fit(self.X_train, self.y_train)
        print("Naive Bayes training complete")
        return self
    
    def evaluate_model(self, save_plot=True):
        """Evaluate the model on the test set and generate ROC curve"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        print("Evaluating model on test set...")
        
        # Get predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        print(f"Test AUC: {roc_auc:.4f}")
        
        # Store results
        self.results['test'] = {
            'accuracy': accuracy,
            'auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr
        }
        
        # Plot ROC curve
        if save_plot:
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {self.task_name} task - Test Set')
            plt.legend(loc="lower right")
            plt.savefig(f'output/{self.task_name}_roc_test.png')
            plt.close()
        
        return self
    
    def cross_validate(self, model_type='svm', kernel='rbf', C=1.0, save_plot=True):
        """Perform k-fold cross-validation"""
        print(f"Performing {self.n_folds}-fold cross-validation...")
        
        # Initialize arrays to store results
        tprs = []
        aucs = []
        accuracies = []
        mean_fpr = np.linspace(0, 1, 100)
        
        # Initialize plot if save_plot is True
        if save_plot:
            plt.figure(figsize=(10, 8))
        
        # Initialize KFold
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=3)
        
        # Perform cross-validation
        for i, (train_idx, test_idx) in enumerate(kf.split(self.X)):
            # Split data
            X_train_cv, X_test_cv = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train_cv, y_test_cv = self.y.iloc[train_idx], self.y.iloc[test_idx]
            
            # Train model
            if model_type == 'svm':
                model = SVC(kernel=kernel, C=C, probability=True, random_state=3)
            elif model_type == 'naive_bayes':
                model = GaussianNB()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            model.fit(X_train_cv, y_train_cv)
            
            # Evaluate model
            y_pred_cv = model.predict(X_test_cv)
            y_pred_prob_cv = model.predict_proba(X_test_cv)[:, 1]
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test_cv, y_pred_cv)
            accuracies.append(accuracy)
            
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_test_cv, y_pred_prob_cv)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            
            # Interpolate TPR at mean FPR points
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            
            # Plot ROC curve for this fold if save_plot is True
            if save_plot:
                plt.plot(fpr, tpr, alpha=0.3, lw=1, label=f'Fold {i+1} (AUC = {roc_auc:.4f})')
            
            print(f"Fold {i+1}: Accuracy = {accuracy:.4f}, AUC = {roc_auc:.4f}")
        
        # Calculate mean and std of TPRs across all folds
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        # Store cross-validation results
        self.results['cv'] = {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'mean_fpr': mean_fpr,
            'mean_tpr': mean_tpr,
            'tprs': tprs,
            'aucs': aucs
        }
        
        print(f"Cross-validation results:")
        print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        
        # Finish and save the plot if save_plot is True
        if save_plot:
            plt.plot(mean_fpr, mean_tpr, 'b-', lw=2,
                    label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves for {self.n_folds}-fold Cross-Validation - {self.task_name} task')
            plt.legend(loc="lower right")
            plt.savefig(f'output/{self.task_name}_roc_cv.png')
            
            # Plot boxplot of AUCs
            plt.figure(figsize=(8, 6))
            plt.boxplot(aucs)
            plt.ylabel('AUC')
            plt.title(f'Distribution of AUC values across {self.n_folds} folds - {self.task_name} task')
            plt.savefig(f'output/{self.task_name}_auc_boxplot.png')
            plt.close()
        
        return self


def main():
    """Main function to run the ML framework"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Machine Learning Framework with Model Training')
    parser.add_argument('task', type=str, choices=['spam', 'p53', 'iris', 'custom'],
                        help='ML task: spam, p53, iris, or custom')
    parser.add_argument('--data', type=str, help='Path to dataset file (required for p53 and custom tasks)')
    parser.add_argument('--model', type=str, choices=['svm', 'naive_bayes'], default='svm',
                        help='Model type: svm or naive_bayes (default: svm)')
    parser.add_argument('--kernel', type=str, choices=['linear', 'poly', 'rbf', 'sigmoid'], default='rbf',
                        help='Kernel for SVM (default: rbf)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of the dataset to include in the test split (default: 0.2)')
    parser.add_argument('--n-folds', type=int, default=10,
                        help='Number of folds for cross-validation (default: 10)')
    parser.add_argument('--no-cv', action='store_true', help='Skip cross-validation')
    parser.add_argument('--no-plots', action='store_true', help='Do not generate plots')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.task in ['p53', 'custom'] and args.data is None:
        parser.error(f"--data is required for {args.task} task")
    
    # Initialize the ML framework
    ml = MLFramework(
        task_name=args.task,
        data_path=args.data,
        test_size=args.test_size,
        n_folds=args.n_folds
    )
    
    # Load data
    ml.load_data()
    
    # Split data
    ml.split_data()
    
    # Train and evaluate model
    if args.model == 'svm':
        ml.train_svm(kernel=args.kernel)
    else:  # naive_bayes
        ml.train_naive_bayes()
    
    ml.evaluate_model(save_plot=not args.no_plots)
    
    # Perform cross-validation if not skipped
    if not args.no_cv:
        ml.cross_validate(model_type=args.model, kernel=args.kernel, save_plot=not args.no_plots)
    
    print("Model training and evaluation complete.")


if __name__ == '__main__':
    main()
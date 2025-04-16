import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(3)

class MLFramework:
    """Main framework class for machine learning tasks"""
    
    def __init__(self, task_name, data_path=None, test_size=0.2):
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
        """
        self.task_name = task_name
        self.data_path = data_path
        self.test_size = test_size
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
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


def main():
    """Main function to run the ML framework"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Basic Machine Learning Framework')
    parser.add_argument('task', type=str, choices=['spam', 'p53', 'iris', 'custom'],
                        help='ML task: spam, p53, iris, or custom')
    parser.add_argument('--data', type=str, help='Path to dataset file (required for p53 and custom tasks)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of the dataset to include in the test split (default: 0.2)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.task in ['p53', 'custom'] and args.data is None:
        parser.error(f"--data is required for {args.task} task")
    
    # Initialize the ML framework
    ml = MLFramework(
        task_name=args.task,
        data_path=args.data,
        test_size=args.test_size
    )
    
    # Load and split data
    ml.load_data()
    ml.split_data()
    
    print("Framework initialized and data prepared successfully.")


if __name__ == '__main__':
    main()
# ML Framework

This project provides a comprehensive Python implementation for hyper specific machine learning classification tasks.

## Overview

The framework includes:

1. A core Machine Learning Framework (`mlFramework.py`) that handles various classification tasks
2. Implementation scripts for:
   - Spam classification with SVM (`p53MutationClassification.py`)
   - P53 mutation classification (`p53MutationClassification.py`)
   - Perceptron and Naive Bayes comparison (`irisClassification.py`)

## Requirements

- Python 3.7 or higher
- Required packages:
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - seaborn

You can install the required packages using pip:

```bash
pip install numpy pandas matplotlib scikit-learn seaborn
```

## Main Framework

The `mlFramework.py` file contains a comprehensive Machine Learning Framework class that supports:

- Loading various datasets (spam, p53, iris, or custom)
- Data preprocessing and validation
- Training SVM and Naive Bayes models
- Evaluating models using accuracy and ROC/AUC metrics
- Cross-validation
- Generating visualizations and reports

### Usage

```python
from ml_framework import MLFramework

# Initialize the framework for a specific task
ml = MLFramework(task_name='spam', n_folds=10)

# Load data, train and evaluate the model
ml.load_data().split_data().train_svm().evaluate_model()

# Perform cross-validation
ml.cross_validate()

# Generate a report
ml.write_report()
```

## Spam Classification

The `spamClassification.py` script implements email classification using Support Vector Machines.

### Features:
- Implements a custom Perceptron algorithm
- Trains a Naive Bayes classifier for comparison
- Evaluates both models on the Iris dataset
- Generates comparison visualizations
- Creates a detailed analysis report

### Usage:

```bash
python irisClassification.py
```

## Using the Framework Directly

The core framework can be used for any classification task:

```python
from ml_framework import MLFramework

# Create framework instance
ml = MLFramework(task_name='custom', data_path='your_data.csv')

# Process pipeline
ml.load_data().split_data().train_svm().evaluate_model()

# Run cross-validation
ml.cross_validate()

# Generate reports and visualizations
ml.write_report()
```

## Command Line Usage

The main framework also supports command line usage:

```bash
python ml_framework.py custom --data your_data.csv --model svm --kernel rbf --n-folds 5
```

## Output

All scripts automatically create an `output` directory containing:
- ROC curve plots
- AUC boxplots
- Confusion matrices
- Performance comparison plots
- Detailed markdown reports

## Customization

The framework is designed to be easily extended:
- Add new model types
- Implement additional metrics
- Support different datasets
- Customize visualizations

## Requirements

```
python >= 3.7
numpy
pandas
matplotlib
scikit-learn
seaborn
```

Install dependencies with:

```bash
pip install -r requirements.txt
```
- Automatically loads the spam dataset
- Splits data into training and test sets
- Trains an SVM model
- Generates ROC curves and calculates AUC
- Performs cross-validation
- Creates visualizations for result analysis

### Usage:

```bash
python p53MutationClassification.py
```

## P53 Mutation Classification

The `p53MutationClassification.py` script analyzes and classifies P53 gene mutations.

### Features:
- Loads P53 mutation data
- Performs cross-validation
- Trains an SVM classifier
- Generates ROC curve visualizations
- Calculates classification metrics

### Usage:

```bash
python p53MutationClassification.py [data_file_path]
```

By default, it looks for `Kato_p53_mutants_200.txt` in the current directory.

## Iris Dataset Classification

The `irisClassification.py` script compares different classification algorithms on the famous Iris dataset.
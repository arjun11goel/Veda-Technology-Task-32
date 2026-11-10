# Task 32: Train-Test Split using Scikit-learn

## Overview
This project demonstrates how to split a machine learning dataset into training and testing subsets using `scikit-learn` in Python. The dataset used for this project is the classic **Iris Dataset**.

---

## Objectives & Deliverables
* **Features (X) & Target (y):** Separated standard feature columns from the classification target variable (`species`).
* **Train-Test Split:** Created training and testing partitions using an **80/20 ratio**.
* **Shape Verification:** Verified data dimensions across both splits.

---

## Tools Used
* **Python**
* **Pandas**
* **Scikit-learn (`sklearn.model_selection.train_test_split`)**
* **Jupyter Notebook**

---

## Execution Output

When performing the 80/20 train-test split on the dataset (150 total samples), the resulting array shapes are:

```text
Training Features Shape (X_train): (120, 4)
Testing Features Shape (X_test): (30, 4)
Training Target Shape (y_train): (120,)
Testing Target Shape (y_test): (30,)
```

---

## How to Run

1. Ensure Python 3.x and the necessary libraries are installed:
   ```bash
   pip install pandas scikit-learn
   ```
2. Place `iris_dataset.csv` in the project root directory.
3. Open and run the notebook or run your script:
   ```python
   import pandas as pd
   from sklearn.model_selection import train_test_split

   df = pd.read_csv('iris_dataset.csv')
   X = df.drop(columns=['species'])
   y = df['species']

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )
   ```

---

## Interview Questions & Answers

### 1. Why do we split data into training and testing sets?
> We split data into separate training and testing sets to evaluate how well a machine learning model generalizes to unseen data. Training a model on all available data can lead to **overfitting**, where the model simply memorizes the dataset rather than learning real underlying patterns.

### 2. What is training data?
> **Training data** is the portion of the dataset used by the machine learning algorithm to learn patterns, correlations, and model weights between the features (X) and target variables (y).

### 3. What is testing data?
> **Testing data** is a distinct, held-out subset of data that the model has never encountered during training. It serves as a final benchmark to measure model performance and accuracy in real-world scenarios.

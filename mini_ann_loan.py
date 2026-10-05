"""
Loan approval prediction with a simple ANN (classification).

Problem: decide whether a loan application should be approved (1) or rejected (0).
Data: loan.csv (Kaggle Loan Prediction). Install: pip install numpy pandas
"""
import numpy as np
import pandas as pd

np.random.seed(5)

# 1. DATASET
data = pd.read_csv("D:\Java\loan.csv").drop(columns=["Loan_ID"], errors="ignore").drop_duplicates()
data = data.dropna(subset=["Loan_Status"]).reset_index(drop=True)
approved = (data["Loan_Status"] == "Y").astype(float).values.reshape(-1, 1)
features = data.drop(columns="Loan_Status")
print(f"Applicants: {len(data)} | Features: {features.shape[1]} | "
      f"Approved: {approved.mean():.0%} | Missing values: {int(features.isna().sum().sum())}")

# 2. PREPROCESSING
rows = np.random.permutation(len(data))
n_train = int(0.8 * len(data))
train_rows = rows[:n_train]
test_rows = rows[n_train:]

for col in features.columns:
    if pd.api.types.is_numeric_dtype(features[col]):
        features[col] = features[col].fillna(features.iloc[train_rows][col].median())
    else:
        features[col] = features[col].fillna(features.iloc[train_rows][col].mode()[0])

features = pd.get_dummies(features, dtype=float)
features = features.loc[:, features.iloc[train_rows].std() > 0]
features = (features - features.iloc[train_rows].mean()) / features.iloc[train_rows].std()

# 3. FEATURE SELECTION
train_features = features.iloc[train_rows]
relation = train_features.corrwith(pd.Series(approved[train_rows, 0], index=train_features.index))
relation = relation.abs().sort_values(ascending=False)

selected = []
for col in relation.index:
    if all(abs(train_features[col].corr(train_features[s])) < 0.85 for s in selected):
        selected.append(col)
    if len(selected) == 5:
        break
print("Selected features:", [f"{col} ({relation[col]:.2f})" for col in selected])

x_train = features[selected].values[train_rows]
x_test = features[selected].values[test_rows]
y_train = approved[train_rows]
y_test = approved[test_rows]

# 4. FUNCTION
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_slope(a):
    return a * (1 - a)

hidden_size = 4
learning_rate = 0.5
epochs = 600

w1 = np.random.randn(x_train.shape[1], hidden_size) * 0.5
b1 = np.zeros((1, hidden_size))
w2 = np.random.randn(hidden_size, 1) * 0.5
b2 = np.zeros((1, 1))

print("\nTraining:")
for epoch in range(1, epochs + 1):
    # Forward propagation
    hidden = sigmoid(x_train @ w1 + b1)
    output = sigmoid(hidden @ w2 + b2)

    error = -np.mean(y_train * np.log(output + 1e-9) + (1 - y_train) * np.log(1 - output + 1e-9))

    # Backward propagation
    d_output = (output - y_train) / len(x_train)
    d_hidden = (d_output @ w2.T) * sigmoid_slope(hidden)

    # Update weights
    w2 -= learning_rate * hidden.T @ d_output
    b2 -= learning_rate * d_output.sum(axis=0, keepdims=True)
    w1 -= learning_rate * x_train.T @ d_hidden
    b1 -= learning_rate * d_hidden.sum(axis=0, keepdims=True)

    if epoch == 1 or epoch % 100 == 0:
        accuracy = np.mean((output >= 0.5) == y_train)
        print(f"  epoch {epoch:3d} | error {error:.4f} | accuracy {accuracy:.2%}")

# 5. PERFORMANCE METRICS
test_output = sigmoid(sigmoid(x_test @ w1 + b1) @ w2 + b2)
predicted = (test_output >= 0.5).astype(int)

tp = int(((predicted == 1) & (y_test == 1)).sum())
tn = int(((predicted == 0) & (y_test == 0)).sum())
fp = int(((predicted == 1) & (y_test == 0)).sum())
fn = int(((predicted == 0) & (y_test == 1)).sum())

accuracy = (tp + tn) / len(y_test)
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
specificity = tn / max(tn + fp, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-9)

print("\nCONFUSION MATRIX (test data)")
print("                    Predicted: Rejected   Predicted: Approved")
print(f"Actual: Rejected    {tn:>12d}   {fp:>18d}")
print(f"Actual: Approved    {fn:>12d}   {tp:>18d}")

print("\nPERFORMANCE METRICS (test data)")
print(f"Accuracy     {accuracy:7.2%}   correct predictions out of all predictions")
print(f"Precision    {precision:7.2%}   of loans we approved, how many should have been approved")
print(f"Recall       {recall:7.2%}   of good applicants, how many we approved")
print(f"Specificity  {specificity:7.2%}   of bad applicants, how many we rejected")
print(f"F1 score     {f1:7.2%}   balance of precision and recall")
print(f"Baseline     {y_test.mean():7.2%}   accuracy if we approved everyone")

train_accuracy = np.mean((sigmoid(sigmoid(x_train @ w1 + b1) @ w2 + b2) >= 0.5) == y_train)
print(f"\nTrain accuracy {train_accuracy:.2%} | Test accuracy {accuracy:.2%}")

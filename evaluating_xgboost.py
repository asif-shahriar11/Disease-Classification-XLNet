import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss, classification_report
import xgboost as xgb

# Assuming 'data/ohsumed/train.csv' and 'data/ohsumed/test.csv' are in a tabular format suitable for XGBoost
# Load data
train_df = pd.read_csv('data/ohsumed/train.csv')
test_df = pd.read_csv('data/ohsumed/test.csv')

# Split the training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_df.drop(columns=['target']), train_df['target'], test_size=0.3, random_state=88)
X_test, y_test = test_df.drop(columns=['target']), test_df['target']

# Convert the datasets into DMatrix objects, which is optimized for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# XGBoost parameters
params = {
    'max_depth': 6,
    'eta': 0.3,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
}
epochs = 10

# Train the model
eval_list = [(dtrain, 'train'), (dval, 'eval')]
bst = xgb.train(params, dtrain, epochs, evals=eval_list)

# Make predictions
preds_val = bst.predict(dval)
preds_test = bst.predict(dtest)

# Convert probabilities to binary output
threshold = 0.5
preds_val_binary = (preds_val > threshold).astype(int)
preds_test_binary = (preds_test > threshold).astype(int)

# Evaluation metrics
accuracy_val = accuracy_score(y_val, preds_val_binary)
f1_val = f1_score(y_val, preds_val_binary)
precision_val = precision_score(y_val, preds_val_binary)
recall_val = recall_score(y_val, preds_val_binary)
hamming_val = hamming_loss(y_val, preds_val_binary)

accuracy_test = accuracy_score(y_test, preds_test_binary)
f1_test = f1_score(y_test, preds_test_binary)
precision_test = precision_score(y_test, preds_test_binary)
recall_test = recall_score(y_test, preds_test_binary)
hamming_test = hamming_loss(y_test, preds_test_binary)

# Print validation metrics
print(f"Validation Accuracy: {accuracy_val}")
print(f"Validation F1 Score: {f1_val}")
print(f"Validation Precision: {precision_val}")
print(f"Validation Recall: {recall_val}")
print(f"Validation Hamming Loss: {hamming_val}")

# Print test metrics
print(f"Test Accuracy: {accuracy_test}")
print(f"Test F1 Score: {f1_test}")
print(f"Test Precision: {precision_test}")
print(f"Test Recall: {recall_test}")
print(f"Test Hamming Loss: {hamming_test}")

# Classification report for validation and test sets
print("\nValidation Classification Report:\n", classification_report(y_val, preds_val_binary))
print("\nTest Classification Report:\n", classification_report(y_test, preds_test_binary))

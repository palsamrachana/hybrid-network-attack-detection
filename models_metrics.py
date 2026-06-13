from models.ml_models import model_manager
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ------------------ Train Models ------------------
X_train, y_train, _ = model_manager.create_sample_training_data(1000)
model_manager.trainer.train_models(X_train, y_train)

# ------------------ Generate Test Data ------------------
X_test, y_test, _ = model_manager.create_sample_training_data(200)
X_test = X_test + np.random.normal(0, 0.3, X_test.shape)

# # ------------------ Load Models ------------------
# model_manager.ensure_models_trained()

rf = model_manager.trainer.models['random_forest']
svm = model_manager.trainer.models['svm']

# ------------------ Predictions ------------------
rf_pred = rf.predict(X_test)
svm_pred = svm.predict(X_test)

# Convert predictions to numeric (0 = normal, 1 = attack)
rf_pred = rf.predict(X_test)
svm_pred = svm.predict(X_test)

# ------------------ PRINT METRICS ------------------

def print_metrics(name, y_true, y_pred):
    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1 Score:", f1_score(y_true, y_pred, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

# Print for both models
print_metrics("Random Forest", y_test, rf_pred)
print_metrics("SVM", y_test, svm_pred)

# ------------------ Accuracy ------------------
rf_acc = accuracy_score(y_test, rf_pred)
svm_acc = accuracy_score(y_test, svm_pred)

plt.figure()
bars = plt.bar(['Random Forest', 'SVM'], [rf_acc, svm_acc])

# Add values on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", ha='center')

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1.1)
plt.show()

# ------------------ Precision / Recall / F1 ------------------
metrics = ['Precision', 'Recall', 'F1 Score']

rf_scores = [
    precision_score(y_test, rf_pred),
    recall_score(y_test, rf_pred),
    f1_score(y_test, rf_pred)
]

svm_scores = [
    precision_score(y_test, svm_pred),
    recall_score(y_test, svm_pred),
    f1_score(y_test, svm_pred)
]

x = np.arange(len(metrics))

plt.figure()
plt.bar(x - 0.2, rf_scores, 0.4, label='Random Forest')
plt.bar(x + 0.2, svm_scores, 0.4, label='SVM')

plt.xticks(x, metrics)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.legend()
plt.show()

# ------------------ Confusion Matrix: Random Forest ------------------
cm_rf = confusion_matrix(y_test, rf_pred)

plt.figure()
sns.heatmap(cm_rf, annot=True, fmt='d',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ------------------ Confusion Matrix: SVM ------------------
cm_svm = confusion_matrix(y_test, svm_pred)

plt.figure()
sns.heatmap(cm_svm, annot=True, fmt='d',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
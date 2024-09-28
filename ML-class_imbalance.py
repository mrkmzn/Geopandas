import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# load data
data = pd.read_csv('bank-full Class Imbalance.csv', sep=';')
# all categorical values to numerical with one-hot encoding
categorical_cols = data.select_dtypes(include=['object']).columns
data1 = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# prepare train and test sets - predict bank client will subscribe term deposit
X = data1.drop(columns='term_dep')
y = data1['term_dep']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25, random_state=42, stratify=y) # sratify on y to manage class imbalance

##########################
# Naive Bayes Classifier: 
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# (TP=-1 TN= 0 FP= 1 FN= 5) as set
cost_matrix = np.array([[-1, 1], [5, 0]])
# confusion matrix
cm = confusion_matrix(y_test, y_pred_nb)
print("Confusion Matrix:\n", cm)
# Total cost
total_cost = (cm[0][1] * cost_matrix[0][1]) + (cm[1][0] * cost_matrix[1][0])
print("Cost for NB:", total_cost)

####################################
# Logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)

# model report
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, y_pred_logistic))

##################################
# model evaluation
# NB
accuracy_bayes = accuracy_score(y_test, y_pred_nb)
precision_bayes = precision_score(y_test, y_pred_nb, pos_label=True)
recall_bayes = recall_score(y_test, y_pred_nb, pos_label=True)
f1_bayes = f1_score(y_test, y_pred_nb, pos_label=True)
confusion_bayes = confusion_matrix(y_test, y_pred_nb)

# LR
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
precision_logistic = precision_score(y_test, y_pred_logistic, pos_label=True)
recall_logistic = recall_score(y_test, y_pred_logistic, pos_label=True)
f1_logistic = f1_score(y_test, y_pred_logistic, pos_label=True)
confusion_logistic = confusion_matrix(y_test, y_pred_logistic)

print("Naive Bayes Metrics:")
print(f"Accuracy: {accuracy_bayes}")
print(f"Precision: {precision_bayes}")
print(f"Recall: {recall_bayes}")
print(f"F1 Score: {f1_bayes}")
print("Confusion Matrix:")
print(confusion_bayes)

print("\nLogistic Regression Metrics:")
print(f"Accuracy: {accuracy_logistic}")
print(f"Precision: {precision_logistic}")
print(f"Recall: {recall_logistic}")
print(f"F1 Score: {f1_logistic}")
print("Confusion Matrix:")
print(confusion_logistic)


# ROC curves
y_prob_bayes = nb_model.predict_proba(X_test) # probability is needed for the curve
y_prob_logistic = logistic_model.predict_proba(X_test)

fp_bayes, tp_bayes, _ = roc_curve(y_test, y_prob_bayes[:, 1], pos_label=True) #false positive rates, true positive rates
roc_auc_bayes = auc(fp_bayes, tp_bayes)

fp_logistic, tp_logistic, _ = roc_curve(y_test, y_prob_logistic[:, 1], pos_label=True)
roc_auc_logistic = auc(fp_logistic, tp_logistic)

# Plot ROC Curve
plt.figure()
plt.plot(fp_bayes, tp_bayes, color='blue', lw=2, label='Naive Bayes (area = {:.2f})'.format(roc_auc_bayes))
plt.plot(fp_logistic, tp_logistic, color='green', lw=2, label='Logistic Regression (area = {:.2f})'.format(roc_auc_logistic))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
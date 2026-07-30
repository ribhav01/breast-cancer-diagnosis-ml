"""
Breast Cancer Diagnosis Prediction
-----------------------------------
Predicts whether a breast tumor is malignant or benign using clinical
measurements from the Wisconsin Breast Cancer dataset (569 patient samples).

Model: Logistic Regression
Result: ~97% test accuracy
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # 0 = malignant, 1 = benign

print("Dataset shape:", df.shape)
print(df['target'].value_counts())

# ---------------------------------------------------------
# 2. Visualize
# ---------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df)
plt.xticks([0, 1], ['Malignant', 'Benign'])
plt.title('Diagnosis Distribution')
plt.savefig('diagnosis_distribution.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.savefig('correlation_heatmap.png', bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 3. Train / test split + scaling
# ---------------------------------------------------------
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. Train model
# ---------------------------------------------------------
model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2%}")
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

# ---------------------------------------------------------
# 5. Feature importance
# ---------------------------------------------------------
coefs = pd.Series(model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
print("\nTop 10 most predictive features:")
print(coefs.head(10))

plt.figure(figsize=(8, 5))
coefs.head(10).plot(kind='barh')
plt.title('Top 10 Predictive Features')
plt.xlabel('Coefficient (impact on prediction)')
plt.gca().invert_yaxis()
plt.savefig('top_features.png', bbox_inches='tight')
plt.close()

print("\nDone. Charts saved as PNG files in this folder.")

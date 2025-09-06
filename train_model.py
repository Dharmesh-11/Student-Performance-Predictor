import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pickle

# ================= Load Dataset =================
df = pd.read_csv("student_dataset_1000.csv")

# ================= Dataset Information =================
print("===== Dataset Information =====")
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn Data Types:")
print(df.dtypes)

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ================= Dataset Visualization =================
# Histogram for numeric columns
df.hist(figsize=(10, 6), bins=20, edgecolor="black")
plt.suptitle("Feature Distributions", fontsize=14)
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# ================= Features and Target =================
X = df[['study_hours', 'attendance', 'sleep_hours', 'midterm_grade']]
y_class = (df['final_grade'] >= 50).astype(int)   # Pass/Fail target
y_reg = df['final_grade']                         # Final grade percentage

# Train-test split
X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

# ================= Feature Scaling =================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================= Logistic Regression (Classification) =================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train_class)

# Predictions
y_pred_class = log_model.predict(X_test_scaled)

# Classification Metrics
accuracy = accuracy_score(y_test_class, y_pred_class) * 100
precision = precision_score(y_test_class, y_pred_class) * 100
recall = recall_score(y_test_class, y_pred_class) * 100
f1 = f1_score(y_test_class, y_pred_class) * 100

print("\n===== Classification Model Performance (Pass/Fail) =====")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1:.2f}%")

# ================= Linear Regression (Percentage Prediction) =================
lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train_reg)

y_pred_reg = lin_model.predict(X_test_scaled)

print("\n===== Regression Model Sample Predictions =====")
print("Actual Grades:    ", list(y_test_reg[:5]))
print("Predicted Grades: ", [round(val, 2) for val in y_pred_reg[:5]])

# ================= Save both models + scaler =================
student_package = {
    "log_model": log_model,
    "lin_model": lin_model,
    "scaler": scaler
}

with open('student_model.pkl', 'wb') as f:
    pickle.dump(student_package, f)

print("\n✅ Logistic Regression + Linear Regression + Scaler saved inside student_model.pkl")

# ================= Visualization =================
# Confusion Matrix
cm = confusion_matrix(y_test_class, y_pred_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Bar chart for metrics
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8,5))
bars = plt.bar(metrics, values, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"])
plt.ylim(0, 100)
plt.ylabel("Percentage")
plt.title("Classification Model Performance (%)")

# Add values on bars
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5, f"{val:.2f}%", 
             ha='center', va='bottom', color='white', fontweight='bold')

plt.show()

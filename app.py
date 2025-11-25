import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    auc
)

# Σταθερά για αναπαραγωγιμότητα
RANDOM_STATE = 42

# =========================
# 1. Φόρτωση και αρχική επισκόπηση του dataset
# =========================

data = pd.read_csv('diabetes_prediction_dataset.csv')

print("Οι πρώτες στήλες του Dataset:")
print(data.head())

print("\nΠληροφορίες για το Dataset:")
print(data.info())

# =========================
# 2. Καθαρισμός δεδομένων
# =========================

print("\nΜοναδικές τιμές στη στήλη gender (πριν την επεξεργασία):")
print(data['gender'].unique())

# Επιτρεπτές τιμές gender: μόνο 'Male' και 'Female'
data['gender'] = data['gender'].apply(
    lambda x: x if x in ['Male', 'Female'] else np.nan
)

# Εύρεση συχνότερης τιμής
most_frequent_gender = data['gender'].mode()[0]
print(f"\nΗ συχνότερη τιμή στο gender που θα χρησιμοποιηθεί για την αντικατάσταση: {most_frequent_gender}")

# Αντικατάσταση μη έγκυρων τιμών gender
data['gender'] = data['gender'].fillna(most_frequent_gender)

print("\nΜοναδικές τιμές στη στήλη gender (μετά την επεξεργασία):")
print(data['gender'].unique())

# Επεξεργασία smoking_history
print("\nΜοναδικές τιμές στη στήλη smoking_history (πριν την επεξεργασία):")
print(data['smoking_history'].unique())

# Αντικατάσταση "No Info" με "none"
data['smoking_history'] = data['smoking_history'].replace("No Info", "none")

# Ενοποίηση κατηγοριών στη στήλη smoking_history
data['smoking_history'] = data['smoking_history'].replace({
    'current': 'smoker',
    'not current': 'non-smoker',
    'former': 'smoker',
    'ever': 'smoker',
    'never': 'non-smoker',
    'none': 'unknown'
})

print("\nΜοναδικές τιμές στη στήλη smoking_history (μετά την ενοποίηση):")
print(data['smoking_history'].unique())

# Έλεγχος βασικών πεδίων
print(f"\nΕύρος ηλικίας: {data['age'].min()} - {data['age'].max()}")
print(f"Μοναδικές τιμές στη στήλη hypertension: {data['hypertension'].unique()}")
print(f"Μοναδικές τιμές στη στήλη heart_disease: {data['heart_disease'].unique()}")
print(f"Εύρος BMI: {data['bmi'].min()} - {data['bmi'].max()}")
print(f"Εύρος HbA1c_level: {data['HbA1c_level'].min()} - {data['HbA1c_level'].max()}")
print(f"Εύρος blood_glucose_level: {data['blood_glucose_level'].min()} - {data['blood_glucose_level'].max()}")
print(f"Μοναδικές τιμές στη στήλη diabetes: {data['diabetes'].unique()}")

# =========================
# 3. Εξερεύνηση δεδομένων (EDA)
# =========================

print("\nΣτατιστικά στοιχεία:")
print(data.describe())

print("\nΚατανομή της μεταβλητής στόχου (diabetes):")
print(data['diabetes'].value_counts())

# =========================
# 4. Κωδικοποίηση κατηγορηματικών & κανονικοποίηση
# =========================

# Label encoding για gender
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])

# One-hot encoding για smoking_history
data = pd.get_dummies(data, columns=['smoking_history'], drop_first=True)

# Κανονικοποίηση αριθμητικών μεταβλητών
scaler = StandardScaler()
columns_to_scale = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

# Διαχωρισμός χαρακτηριστικών και μεταβλητής στόχου
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Διαχωρισμός σε train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print("\nΜέγεθος Συνόλου Εκπαίδευσης:", X_train.shape)
print("Μέγεθος Συνόλου Δοκιμής:", X_test.shape)
print("\nΗ προ-επεξεργασία ολοκληρώθηκε!")

print("\nΣτατιστική περιγραφή αριθμητικών στηλών:")
print(data.describe())

# =========================
# 5. Γραφήματα κατανομής & συσχέτισης (non-blocking)
# =========================

# Κατανομή ηλικίας
plt.figure(figsize=(8, 5))
sns.histplot(data['age'], kde=True, bins=30)
plt.title('Κατανομή Ηλικίας')
plt.xlabel('Ηλικία')
plt.ylabel('Συχνότητα')
plt.show(block=False)

# Κατανομή BMI
plt.figure(figsize=(8, 5))
sns.histplot(data['bmi'], kde=True, bins=30)
plt.title('Κατανομή ΔΜΣ (BMI)')
plt.xlabel('ΔΜΣ')
plt.ylabel('Συχνότητα')
plt.show(block=False)

# Κατανομή HbA1c_level
plt.figure(figsize=(8, 5))
sns.histplot(data['HbA1c_level'], kde=True, bins=30)
plt.title('Κατανομή Επίπεδου HbA1c')
plt.xlabel('Επίπεδο HbA1c')
plt.ylabel('Συχνότητα')
plt.show(block=False)

# Κατανομή blood_glucose_level
plt.figure(figsize=(8, 5))
sns.histplot(data['blood_glucose_level'], kde=True, bins=30)
plt.title('Κατανομή Επίπεδου Γλυκόζης στο Αίμα')
plt.xlabel('Επίπεδο Γλυκόζης')
plt.ylabel('Συχνότητα')
plt.show(block=False)

# Κατανομή μεταβλητής στόχου
plt.figure(figsize=(8, 5))
sns.countplot(x='diabetes', data=data)
plt.title('Κατανομή της Μεταβλητής Στόχου (Diabetes)')
plt.xlabel('Diabetes (0: Όχι, 1: Ναι)')
plt.ylabel('Συχνότητα')
plt.show(block=False)

# Πίνακας συσχέτισης
correlation_matrix = data.corr()
print("\nΠίνακας Συσχέτισης:")
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Heatmap Πίνακα Συσχέτισης')
plt.show(block=False)

# =========================
# 6. Εκπαίδευση βασικών μοντέλων (Logistic Regression, Random Forest, SVM)
# =========================

logreg = LogisticRegression(random_state=RANDOM_STATE, max_iter=500)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

rf = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

svm = SVC(random_state=RANDOM_STATE)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

models = {
    "Logistic Regression": y_pred_logreg,
    "Random Forest": y_pred_rf,
    "SVM": y_pred_svm
}

for model_name, y_pred in models.items():
    print(f"\n{model_name}:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Πίνακας Σύγχυσης: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show(block=False)

# =========================
# 7. SMOTE + Random Forest
# =========================

smote = SMOTE(random_state=RANDOM_STATE)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

rf_smote = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100)
rf_smote.fit(X_resampled, y_resampled)

y_pred_rf_smote = rf_smote.predict(X_test)

print("Classification Report (Random Forest with SMOTE):")
print(classification_report(y_test, y_pred_rf_smote))

cm = confusion_matrix(y_test, y_pred_rf_smote)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Πίνακας Σύγχυσης: Random Forest with SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show(block=False)

# =========================
# 8. Randomized Search σε 10% των resampled δεδομένων (Random Forest)
# =========================

X_sample, _, y_sample, _ = train_test_split(
    X_resampled, y_resampled, test_size=0.9, random_state=RANDOM_STATE
)

print(f"Μέγεθος δείγματος για Grid Search: {X_sample.shape[0]} δείγματα")

param_distributions = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE),
    param_distributions=param_distributions,
    n_iter=10,
    cv=2,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=2
)
random_search.fit(X_sample, y_sample)

print("Καλύτερες Υπερπαράμετροι:", random_search.best_params_)

best_rf = random_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

print("Classification Report (Optimized Random Forest):")
print(classification_report(y_test, y_pred_best_rf))

cm = confusion_matrix(y_test, y_pred_best_rf)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Πίνακας Σύγχυσης: Optimized Random Forest (10% του Dataset)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show(block=False)

# Σημασία χαρακτηριστικών
feature_importances = best_rf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Σημασία Χαρακτηριστικών (Feature Importance)')
plt.xlabel('Σημασία')
plt.ylabel('Χαρακτηριστικό')
plt.show(block=False)

# =========================
# 9. Εκπαίδευση XGBoost
# =========================

xgb = XGBClassifier(
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print("Classification Report (XGBoost):")
print(classification_report(y_test, y_pred_xgb))

cm = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Πίνακας Σύγχυσης: XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show(block=False)

# =========================
# 10. Εκπαίδευση LightGBM
# =========================

lgbm = LGBMClassifier(random_state=RANDOM_STATE)
lgbm.fit(X_train, y_train)

y_pred_lgbm = lgbm.predict(X_test)

print("Classification Report (LightGBM):")
print(classification_report(y_test, y_pred_lgbm))

cm = confusion_matrix(y_test, y_pred_lgbm)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Πίνακας Σύγχυσης: LightGBM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show(block=False)

# =========================
# 11. ROC & AUC για Random Forest, XGBoost, LightGBM
# =========================

y_prob_rf = best_rf.predict_proba(X_test)[:, 1]
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
y_prob_lgbm = lgbm.predict_proba(X_test)[:, 1]

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
fpr_lgbm, tpr_lgbm, _ = roc_curve(y_test, y_prob_lgbm)

roc_auc_rf = auc(fpr_rf, tpr_rf)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
roc_auc_lgbm = auc(fpr_lgbm, tpr_lgbm)

plt.figure(figsize=(10, 8))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf:.2f})", linestyle='--')
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {roc_auc_xgb:.2f})", linestyle='-.')
plt.plot(fpr_lgbm, tpr_lgbm, label=f"LightGBM (AUC = {roc_auc_lgbm:.2f})", linestyle='-')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label="Random Guess")
plt.title("Καμπύλη ROC και AUC για Ταξινομητές", fontsize=14)
plt.xlabel("False Positive Rate (FPR)", fontsize=12)
plt.ylabel("True Positive Rate (TPR)", fontsize=12)
plt.legend(loc="lower right")
plt.grid()
plt.show(block=False)

# =========================
# 12. Αποθήκευση και φόρτωση μοντέλου LightGBM
# =========================

model_filename = 'lightgbm_model.pkl'
joblib.dump(lgbm, model_filename)
print(f"Το μοντέλο LightGBM αποθηκεύτηκε ως: {model_filename}")

loaded_model = joblib.load(model_filename)
print("Το μοντέλο LightGBM φορτώθηκε επιτυχώς.")

y_pred_loaded = loaded_model.predict(X_test)

print("Classification Report (Loaded Model):")
print(classification_report(y_test, y_pred_loaded))

# Για να μη κλείνει αμέσως το script και τα plots:
input("\nΤέλος εκτέλεσης. Πάτησε Enter για έξοδο...")

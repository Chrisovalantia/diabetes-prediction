# Πρόβλεψη Διαβήτη με Μηχανική Μάθηση  
*Διπλωματική Εργασία – Αναλυτική Τεκμηρίωση Project*

---

## Περιεχόμενα

- [Εισαγωγή](#-εισαγωγή)
- [Τεχνολογίες που χρησιμοποιούνται](#-τεχνολογίες-που-χρησιμοποιούνται)
- [Δομή του Project](#-δομή-του-project)
- [Εγκατάσταση](#-εγκατάσταση)
- [Εκτέλεση του Κώδικα](#-εκτέλεση-του-κώδικα)
- [Αναλυτική Περιγραφή των Βημάτων](#-αναλυτική-περιγραφή-των-βημάτων)
  - [1. Φόρτωση & Καθαρισμός Δεδομένων](#1️⃣-φόρτωση--καθαρισμός-δεδομένων)
  - [2. Προεπεξεργασία](#2️⃣-προεπεξεργασία)
  - [3. EDA – Εξερευνητική Ανάλυση](#3️⃣-eda--εξερευνητική-ανάλυση)
  - [4. Εκπαίδευση Μοντέλων](#4️⃣-εκπαίδευση-μοντέλων)
  - [5. SMOTE Oversampling](#5️⃣-smote-oversampling)
  - [6. Βελτιστοποίηση Random Forest](#6️⃣-βελτιστοποίηση-random-forest)
  - [7. XGBoost & LightGBM](#7️⃣-xgboost--lightgbm)
  - [8. ROC Curves](#8️⃣-roc-curves)
  - [9. Αποθήκευση Μοντέλου](#9️⃣-αποθήκευση-μοντέλου)
- [Αρχεία του Project](#-αρχεία-του-project)
- [requirements.txt](#-requirementstxt)
- [Μελλοντικές Επεκτάσεις](#-μελλοντικές-επεκτάσεις)

---

# Εισαγωγή

Το project αυτό υλοποιεί ένα πλήρες pipeline μηχανικής μάθησης για την **πρόβλεψη διαβήτη** χρησιμοποιώντας πραγματικά δεδομένα ασθενών.

Στόχοι:

- Καθαρισμός και προεπεξεργασία δεδομένων  
- Εξερευνητική ανάλυση (EDA) με γραφήματα  
- Εκπαίδευση πολλαπλών μοντέλων  
- Αντιμετώπιση ανισορροπίας δεδομένων  
- Βελτιστοποίηση υπερπαραμέτρων  
- Σύγκριση μοντέλων μέσω ROC & AUC  
- Αποθήκευση τελικού μοντέλου  

Η έμφαση δίνεται στην ακρίβεια, αξιοπιστία και τεκμηρίωση του συστήματος.

---

# Τεχνολογίες που Χρησιμοποιούνται

- **Python 3**
- **NumPy**, **Pandas** – επεξεργασία δεδομένων
- **Matplotlib**, **Seaborn** – γραφήματα
- **Scikit-Learn** – βασικά μοντέλα και pre-processing
- **Imbalanced-Learn (SMOTE)** – αντιμετώπιση ανισορροπίας
- **XGBoost**
- **LightGBM**
- **Joblib** – αποθήκευση μοντέλου

---

# Δομή του Project

```text
diabetes-prediction/
├─ app.py
├─ diabetes_prediction_dataset.csv
├─ requirements.txt
└─ venv/  (virtual environment)
```

---

# Εγκατάσταση

### Δημιουργία virtual environment (προαιρετικό αλλά προτείνεται)

```bash
python3 -m venv venv
source venv/bin/activate     # macOS / Linux
```

### Εγκατάσταση εξαρτήσεων

```bash
pip install -r requirements.txt
```

---

# Εκτέλεση του Κώδικα

Βεβαιώσου ότι το dataset βρίσκεται στον ίδιο φάκελο με το `app.py`.

Τρέξε:

```bash
python3 app.py
```

Ο κώδικας θα:

- φορτώσει και καθαρίσει τα δεδομένα,
- εκτελέσει την EDA,
- εκπαιδεύσει όλα τα μοντέλα,
- εμφανίσει όλα τα plots **ταυτόχρονα** (επειδή χρησιμοποιείται `plt.show(block=False)`)

---

# Αναλυτική Περιγραφή των Βημάτων

## Φόρτωση & Καθαρισμός Δεδομένων

- Φόρτωση CSV με `pandas`.
- Έλεγχος τύπων και δομής των δεδομένων.
- Διόρθωση της στήλης **gender**: 
  - επιτρεπτές τιμές μόνο `Male`, `Female`.
- Ενοποίηση κατηγοριών **smoking_history**:
  - μετατροπές: `current`, `former`, `ever` → `smoker`
  - `never`, `not current` → `non-smoker`
  - `No Info` → `unknown`.

## Προεπεξεργασία

- Label Encoding στη στήλη **gender**.
- One-Hot Encoding στη **smoking_history**.
- Κανονικοποίηση των μεταβλητών:
  - `age`, `bmi`, `HbA1c_level`, `blood_glucose_level`.
- Διαχωρισμός σε Train/Test με 80/20 split.

## EDA – Εξερευνητική Ανάλυση

Παράγονται αυτόματα:

- Histogram ηλικίας  
- Histogram BMI  
- Histogram HbA1c  
- Histogram glucose  
- Bar plot μεταβλητής στόχου  
- Heatmap συσχέτισης χαρακτηριστικών  

Όλα εμφανίζονται ταυτόχρονα.

## Εκπαίδευση Μοντέλων

Εκπαιδεύονται:

- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**

Για κάθε μοντέλο εμφανίζονται:

- Accuracy  
- Classification Report (Precision, Recall, F1-score)  
- Confusion Matrix (Heatmap)  

## SMOTE Oversampling

Λόγω ανισορροπίας στόχου, εφαρμόζεται **SMOTE** στο training set.

Γίνεται εκπαίδευση Random Forest με τα ενισχυμένα δεδομένα  
και εμφανίζεται νέο confusion matrix.

## Βελτιστοποίηση Random Forest

Με χρήση:

- `RandomizedSearchCV`  
- Μόνο σε **10% των δεδομένων** για ταχύτητα  

Επιστρέφονται:

- Βέλτιστες υπερπαράμετροι  
- Νέο confusion matrix  
- Σημασία χαρακτηριστικών (feature importances)  

## XGBoost & LightGBM

Εκπαίδευση δύο gradient boosting μοντέλων:

- **XGBoost**  
- **LightGBM**

Και για τα δύο, προβάλλονται:

- Classification Report  
- Confusion Matrix  

## ROC Curves

Συγκρίνονται:

- Optimized Random Forest  
- XGBoost  
- LightGBM  

Στο ίδιο διάγραμμα εμφανίζονται όλες οι ROC καμπύλες και οι AUC τιμές.

## Αποθήκευση Μοντέλου

Το **LightGBM** αποθηκεύεται σε αρχείο:

```text
lightgbm_model.pkl
```

Και φορτώνεται ξανά για επιβεβαίωση ορθής λειτουργίας.

---

# Αρχεία του Project

### `app.py`
Πλήρης υλοποίηση όλης της ροής.

### `diabetes_prediction_dataset.csv`
Το dataset εκπαίδευσης.

### `lightgbm_model.pkl`
Το τελικό εκπαιδευμένο μοντέλο.

### `requirements.txt`
Ο κατάλογος εξαρτήσεων (βλ. επόμενο τμήμα).

---

# requirements.txt

Αυτό είναι το **τελικό και πλήρες** requirements file για το project:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
lightgbm
joblib
```

---

# Πηγή του dataset

To dataset λήφθηκε από το kaggle με τίτλο [Diabetes prediction dataset
](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset?resource=download)

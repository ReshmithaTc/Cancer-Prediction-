import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import pickle

# ===================== LOAD DATA =====================
file_path = "AI_Brosnan_CancerDataset.xlsx"
df = pd.read_excel("AI_Brosnan_CancerDataset.xlsx", sheet_name="Cleaned_Data")
print("Dataset Loaded:", df.shape)

# ===================== FEATURE ENGINEERING =====================
df_model = df.drop(columns=["Name"]).copy()

risk_threshold = df_model["Risk_Score"].median()
df_model["High_Risk"] = (df_model["Risk_Score"] > risk_threshold).astype(int)


ordinal_categories = [ 
    "Alcohol_Consumption",  
    "Insurance_Type"  
]

# Define categories without ranking (nominal) - use One-Hot Encoding
nominal_categories = [
    "Gender", 
    "Smoking",  
    "Diabetes",  
    "Hypertension",  
    "Heart_Disease",  
]

# Check which columns actually exist in our dataframe
available_ordinal = [col for col in ordinal_categories if col in df_model.columns]
available_nominal = [col for col in nominal_categories if col in df_model.columns]

print(f"\nEncoding Strategy:")
print(f"Ordinal categories (Label Encoding): {available_ordinal}")
print(f"Nominal categories (One-Hot Encoding): {available_nominal}")

# 1. LABEL ENCODING for ordinal categories
label_encoders = {}
for col in available_ordinal:
    le = LabelEncoder()
    df_model[col + "_encoded"] = le.fit_transform(df_model[col])
    label_encoders[col] = le
    print(f"  {col} encoded as: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 2. ONE-HOT ENCODING for nominal categories
if available_nominal:
    df_ohe = pd.get_dummies(df_model[available_nominal], 
                           prefix=available_nominal,
                           drop_first=True)
    print(f"\nCreated {df_ohe.shape[1]} one-hot encoded features")
else:
    df_ohe = pd.DataFrame()  # Empty dataframe if no nominal columns

# ===================== NUMERICAL FEATURES =====================
numerical_features = [
    "Age", "Height_cm", "Weight_kg", "BMI", "Systolic_BP", "Diastolic_BP",
    "Heart_Rate", "Temperature_F", "Blood_Sugar", "Cholesterol",
    "Hemoglobin", "Exercise_Hours_Week", "Hospital_Visits_Year"
]

# ===================== ADDITIONAL FEATURE ENGINEERING =====================
df_model["BMI_Age_Interaction"] = df_model["BMI"] * df_model["Age"]
df_model["BP_Ratio"] = df_model["Systolic_BP"] / df_model["Diastolic_BP"].replace(0, np.nan)
df_model["BP_Ratio"].fillna(df_model["BP_Ratio"].median(), inplace=True)

df_model["High_Cholesterol"] = (df_model["Cholesterol"] > df_model["Cholesterol"].median()).astype(int)
df_model["High_Blood_Sugar"] = (df_model["Blood_Sugar"] > df_model["Blood_Sugar"].median()).astype(int)
df_model["Low_Exercise"] = (df_model["Exercise_Hours_Week"] < df_model["Exercise_Hours_Week"].median()).astype(int)

df_model["Health_Score"] = (
    (df_model["Exercise_Hours_Week"] / df_model["Exercise_Hours_Week"].max()) * 0.3 +
    (1 - df_model["BMI"] / df_model["BMI"].max()) * 0.3 +
    (1 - df_model["Cholesterol"] / df_model["Cholesterol"].max()) * 0.2 +
    (1 - df_model["Blood_Sugar"] / df_model["Blood_Sugar"].max()) * 0.2
)

engineered_features = [
    "BMI_Age_Interaction", "BP_Ratio", "Health_Score",
    "High_Cholesterol", "High_Blood_Sugar", "Low_Exercise"
]

numerical_features.extend(engineered_features)

# ===================== CREATE FINAL FEATURE MATRIX =====================
# Collect all feature components
feature_components = []

# 1. Numerical features
feature_components.append(df_model[numerical_features])

# 2. Label encoded ordinal features
if available_ordinal:
    label_encoded_features = [col + "_encoded" for col in available_ordinal]
    feature_components.append(df_model[label_encoded_features])

# 3. One-hot encoded nominal features
if not df_ohe.empty:
    feature_components.append(df_ohe)

# Combine all features
if len(feature_components) > 1:
    X = pd.concat(feature_components, axis=1)
else:
    X = feature_components[0]

y = df_model["High_Risk"]

print(f"\nFeature Engineering Completed")
print(f"Total Features Used: {X.shape[1]}")
print(f"Target distribution: {dict(y.value_counts())}")
print(f"\nFeature breakdown:")
print(f"- Numerical features: {len(numerical_features)}")
print(f"- Label encoded features: {len(available_ordinal)}")
print(f"- One-hot encoded features: {df_ohe.shape[1] if not df_ohe.empty else 0}")
print(f"\nFirst few features: {list(X.columns[:10])}")


# ===================== TRAIN TEST SPLIT & SCALING =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

X_train_scaled.to_csv("X_train.csv", index=False)
X_test_scaled.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

total = len(X)
split_info = {
    "train_size": len(X_train),
    "test_size": len(X_test),
    "train_ratio": len(X_train) / total,
    "test_ratio": len(X_test) / total,
    "train_target_dist": y_train.value_counts().to_dict(),
    "test_target_dist": y_test.value_counts().to_dict()
}

with open("split_info.pkl", "wb") as f:
    pickle.dump(split_info, f)

print(f"\nTraining set: {split_info['train_size']} samples ({split_info['train_ratio']:.2%})")
print(f"Test set: {split_info['test_size']} samples ({split_info['test_ratio']:.2%})")
print(f"Train target distribution: {split_info['train_target_dist']}")
print(f"Test target distribution: {split_info['test_target_dist']}")
print("Saved objects: scaler.pkl, split_info.pkl")


# ===================== MULTI-MODEL TRAINING & SELECTION =====================
def train_multiple_models(X_train, y_train):
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    trained_models = {}
    model_scores = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)
        trained_models[name] = model

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

        model_scores[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }

        print(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    return trained_models, model_scores


def select_best_model(model_scores):
    best_model_name = max(model_scores, key=lambda x: model_scores[x]['cv_mean'])
    best_score = model_scores[best_model_name]['cv_mean']
    return best_model_name, best_score


def save_models(trained_models, model_scores, best_model_name):
    with open('trained_models.pkl', 'wb') as f:
        pickle.dump(trained_models, f)

    with open('model_scores.pkl', 'wb') as f:
        pickle.dump(model_scores, f)

    with open('best_model.pkl', 'wb') as f:
        pickle.dump(trained_models[best_model_name], f)

    model_info = {
        'best_model_name': best_model_name,
        'best_model_score': model_scores[best_model_name]['cv_mean'],
        'all_model_scores': {name: scores['cv_mean'] for name, scores in model_scores.items()}
    }

    with open('model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)

    return model_info


# ===================== RUN TRAINING =====================
print("\n===================== TRAINING MODELS =====================")

trained_models, model_scores = train_multiple_models(X_train_scaled, y_train)

best_model_name, best_score = select_best_model(model_scores)

model_info = save_models(trained_models, model_scores, best_model_name)

print("\n===================== MODEL SELECTION RESULTS =====================")
print(f"Best model: {best_model_name}")
print(f"Best CV score: {best_score:.4f}")

print("\nAll model scores:")
for name, score in model_info['all_model_scores'].items():
    print(f"  {name}: {score:.4f}")



# ===================== MODEL EVALUATION CODE (UNCHANGED) =====================
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import pickle

def evaluate_all_models(trained_models, X_test, y_test):
    results = {}

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)

        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }

    return results


def create_main_dashboard(results, y_test):
    model_names = list(results.keys())

    accuracies = [results[m]["accuracy"] for m in model_names]
    precisions = [results[m]["precision"] for m in model_names]
    recalls = [results[m]["recall"] for m in model_names]
    f1s = [results[m]["f1"] for m in model_names]
    aucs = [results[m]["auc"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.2

    plt.figure(figsize=(18, 10))

    # PERFORMANCE COMPARISON
    plt.subplot(2, 2, 1)
    plt.bar(x - 1.5*width, accuracies, width, label="Accuracy")
    plt.bar(x - 0.5*width, precisions, width, label="Precision")
    plt.bar(x + 0.5*width, recalls, width, label="Recall")
    plt.bar(x + 1.5*width, f1s, width, label="F1")
    plt.xticks(x, model_names, rotation=45)
    plt.title("Model Performance Comparison")
    plt.legend()

    # ROC CURVES
    plt.subplot(2, 2, 2)
    for m in model_names:
        if results[m]["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, results[m]["y_proba"])
            plt.plot(fpr, tpr, label=f"{m} (AUC={results[m]['auc']:.3f})")
    plt.legend()
    plt.title("ROC Curves")

    plt.tight_layout()
    plt.savefig("model_evaluation_dashboard.png", dpi=300)
    plt.show()


def plot_all_confusion_matrices(results):
    for model_name, data in results.items():
        plt.figure(figsize=(4, 4))
        sns.heatmap(data["confusion_matrix"], annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.show()


def save_evaluation_results(results):
    summary = []

    for model, m in results.items():
        summary.append({
            "Model": model,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "AUC": m["auc"]
        })

    df = pd.DataFrame(summary)
    df.to_csv("model_evaluation_summary.csv", index=False)

    with open("evaluation_results.pkl", "wb") as f:
        pickle.dump(results, f)

    return df


if __name__ == "__main__":

    with open("trained_models.pkl", "rb") as f:
        trained_models_loaded = pickle.load(f)

    X_test_loaded = pd.read_csv("X_test.csv")
    y_test_loaded = pd.read_csv("y_test.csv").squeeze()

    results = evaluate_all_models(trained_models_loaded, X_test_loaded, y_test_loaded)

    create_main_dashboard(results, y_test_loaded)

    plot_all_confusion_matrices(results)

    summary_df = save_evaluation_results(results)

    print("\nModel Evaluation Summary:")
    print(summary_df)



# =============================================================
#                   ★★★ INFERENCE PIPELINE ★★★
# =============================================================

print("\n========= INFERENCE MODULE LOADED =========")

# Load required saved files
best_model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
trained_models_all = pickle.load(open("trained_models.pkl", "rb"))

# LABEL ENCODERS: extracted from training
label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    label_encoders[col].fit(df[col])  # fit using training categories


# ---------------- PREPROCESS FUNCTION ----------------
def preprocess_new_patient(patient_dict):

    df_new = pd.DataFrame([patient_dict])

    # Apply label encoding (safe version)
    for col in categorical_columns:
        try:
            df_new[col + "_encoded"] = label_encoders[col].transform(df_new[col])
        except:
            df_new[col + "_encoded"] = 0

    # OHE for inference
    df_ohe_new = pd.get_dummies(df_new[categorical_columns], drop_first=True)
    for col in df_ohe.columns:
        if col not in df_ohe_new:
            df_ohe_new[col] = 0
    df_ohe_new = df_ohe_new[df_ohe.columns]

    # Engineered features
    df_new["BMI_Age_Interaction"] = df_new["BMI"] * df_new["Age"]

    df_new["BP_Ratio"] = df_new["Systolic_BP"] / df_new["Diastolic_BP"].replace(0, np.nan)
    df_new["BP_Ratio"].fillna(df_model["BP_Ratio"].median(), inplace=True)

    df_new["High_Cholesterol"] = (
        1 if df_new["Cholesterol"].iloc[0] > df_model["Cholesterol"].median() else 0
    )
    df_new["High_Blood_Sugar"] = (
        1 if df_new["Blood_Sugar"].iloc[0] > df_model["Blood_Sugar"].median() else 0
    )
    df_new["Low_Exercise"] = (
        1 if df_new["Exercise_Hours_Week"].iloc[0] < df_model["Exercise_Hours_Week"].median() else 0
    )

    df_new["Health_Score"] = (
        (df_new["Exercise_Hours_Week"] / df_model["Exercise_Hours_Week"].max()) * 0.3 +
        (1 - df_new["BMI"] / df_model["BMI"].max()) * 0.3 +
        (1 - df_new["Cholesterol"] / df_model["Cholesterol"].max()) * 0.2 +
        (1 - df_new["Blood_Sugar"] / df_model["Blood_Sugar"].max()) * 0.2
    )

    # Build final feature vector
    final_df = pd.concat([
        df_new[numerical_features],
        df_new[[col + "_encoded" for col in categorical_columns]],
        df_ohe_new
    ], axis=1)

    final_df = final_df.reindex(columns=X.columns, fill_value=0)

    scaled = scaler.transform(final_df)
    return scaled


# ------------------------- PREDICT FUNCTION ---------------------------
def predict_patient(data):
    processed = preprocess_new_patient(data)
    pred = best_model.predict(processed)[0]
    prob = best_model.predict_proba(processed)[0][1]
    return {
        "Prediction": "HIGH RISK" if pred == 1 else "LOW RISK",
        "Probability": float(round(prob, 4))
    }


# ------------------------- SAMPLE TEST -------------------------------
sample = {
    "Age": 45,
    "Height_cm": 170,
    "Weight_kg": 75,
    "BMI": 75 / (1.7**2),
    "Systolic_BP": 135,
    "Diastolic_BP": 85,
    "Heart_Rate": 80,
    "Temperature_F": 98.6,
    "Blood_Sugar": 120,
    "Cholesterol": 210,
    "Hemoglobin": 14,
    "Exercise_Hours_Week": 2,
    "Hospital_Visits_Year": 1,

    "Gender": "Female",
    "Smoking": "No",
    "Alcohol_Consumption": "No",
    "Diabetes": "No",
    "Hypertension": "Yes",
    "Heart_Disease": "No",
    "Insurance_Type": "Public"
}

print("\n===== SAMPLE INFERENCE RESULT =====")
print(predict_patient(sample))
# =============================================================
#                 DISPLAY STYLE OUTPUT
# =============================================================

if __name__ == "__main__":

    print("Model Inference System")
    print("=" * 50)

    print("Single Patient Prediction Demo:")
    print("-" * 40)

    # Show input
    print("Patient Data:")
    for k, v in sample.items():
        print(f"  {k}: {v}")

    # Call your existing function
    result = predict_patient(sample)

    print("\nPrediction Results:")
    print(f"  Risk Level: {result['Prediction']}")
    print(f"  Probability: {result['Probability']}")

    print("\nTo use this system in your own code:")
    print("from model_inference import ModelInference")
    print("inference = ModelInference()")
    print("result = inference.predict_single(patient_data)")

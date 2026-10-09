"""
Retrain heart / stroke / hepatitis models from raw CSVs.
- Scales numeric features inside a saved bundle (fixes raw-vs-scaled mismatch)
- Uses SMOTE on the training split for imbalanced targets
"""
from __future__ import annotations

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from model_bundle import DiseaseModel

warnings.filterwarnings("ignore")

OUT_DIR = "Trained_Models"
os.makedirs(OUT_DIR, exist_ok=True)


def _fit_bundle(
    X,
    y,
    numeric_cols,
    model,
    use_smote=True,
    smote_ratio=1.0,
    calibrate=False,
    threshold=0.5,
):
    feature_order = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = X_train.copy()
    X_test_s = X_test.copy()
    if numeric_cols:
        X_train_s[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_s[numeric_cols] = scaler.transform(X_test[numeric_cols])

    if use_smote:
        minority = int((y_train == 1).sum())
        k = max(1, min(5, minority - 1))
        sm = SMOTE(random_state=42, k_neighbors=k, sampling_strategy=smote_ratio)
        X_res, y_res = sm.fit_resample(X_train_s, y_train)
        print(f"  SMOTE: {np.bincount(y_train)} -> {np.bincount(y_res)}")
    else:
        X_res, y_res = X_train_s, y_train

    if calibrate:
        model = CalibratedClassifierCV(model, cv=3, method="isotonic")

    model.fit(X_res, y_res)
    proba = model.predict_proba(X_test_s)[:, 1]
    pred = (proba >= threshold).astype(int)

    print(classification_report(y_test, pred, digits=3))
    try:
        print(f"  ROC-AUC: {roc_auc_score(y_test, proba):.4f}")
    except ValueError:
        pass

    return DiseaseModel(model, scaler, numeric_cols, feature_order, threshold)


def prepare_heart():
    df = pd.read_csv("Datasets/heart.csv")
    df["Sex"] = (df["Sex"] == "M").astype(int)
    df["ExerciseAngina_Y"] = (df["ExerciseAngina"] == "Y").astype(int)

    cp = pd.get_dummies(df["ChestPainType"], prefix="ChestPainType")
    # Drop ASY as reference (matches existing clean schema)
    for col in ["ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA"]:
        if col not in cp.columns:
            cp[col] = 0
    cp = cp[["ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA"]]

    ecg = pd.get_dummies(df["RestingECG"], prefix="RestingECG")
    for col in ["RestingECG_Normal", "RestingECG_ST"]:
        if col not in ecg.columns:
            ecg[col] = 0
    ecg = ecg[["RestingECG_Normal", "RestingECG_ST"]]

    slope = pd.get_dummies(df["ST_Slope"], prefix="ST_Slope")
    for col in ["ST_Slope_Flat", "ST_Slope_Up"]:
        if col not in slope.columns:
            slope[col] = 0
    slope = slope[["ST_Slope_Flat", "ST_Slope_Up"]]

    X = pd.concat(
        [
            df[["Age", "Sex", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]],
            cp,
            ecg,
            df[["ExerciseAngina_Y"]],
            slope,
        ],
        axis=1,
    )
    y = df["HeartDisease"].astype(int)
    numeric = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    return X, y, numeric


def prepare_stroke():
    df = pd.read_csv("Datasets/stroke.csv")
    df = df.drop(columns=["id"], errors="ignore")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    df["ever_married"] = (df["ever_married"] == "Yes").astype(int)

    gender = pd.get_dummies(df["gender"], prefix="gender")
    for col in ["gender_Male", "gender_Other"]:
        if col not in gender.columns:
            gender[col] = 0
    gender = gender[["gender_Male", "gender_Other"]]

    work = pd.get_dummies(df["work_type"], prefix="work_type")
    work_cols = [
        "work_type_Govt_job",
        "work_type_Never_worked",
        "work_type_Private",
        "work_type_Self-employed",
        "work_type_children",
    ]
    for col in work_cols:
        if col not in work.columns:
            work[col] = 0
    work = work[work_cols]

    res = pd.get_dummies(df["Residence_type"], prefix="Residence_type")
    for col in ["Residence_type_Rural", "Residence_type_Urban"]:
        if col not in res.columns:
            res[col] = 0
    res = res[["Residence_type_Rural", "Residence_type_Urban"]]

    smoke = pd.get_dummies(df["smoking_status"], prefix="smoking_status")
    smoke_cols = [
        "smoking_status_Unknown",
        "smoking_status_formerly smoked",
        "smoking_status_never smoked",
        "smoking_status_smokes",
    ]
    for col in smoke_cols:
        if col not in smoke.columns:
            smoke[col] = 0
    smoke = smoke[smoke_cols]

    X = pd.concat(
        [
            df[["age", "hypertension", "heart_disease", "ever_married", "avg_glucose_level", "bmi"]],
            gender,
            work,
            res,
            smoke,
        ],
        axis=1,
    )
    y = df["stroke"].astype(int)
    numeric = ["age", "avg_glucose_level", "bmi"]
    return X, y, numeric


def prepare_hepatitis():
    df = pd.read_csv("Datasets/hepatitis.csv")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    # Binary: blood donors (incl. suspect) = 0, hepatitis/fibrosis/cirrhosis = 1
    healthy = {"0=Blood Donor", "0s=suspect Blood Donor"}
    df["target"] = (~df["Category"].isin(healthy)).astype(int)
    df["Sex"] = (df["Sex"].str.lower() == "m").astype(int)

    feature_cols = [
        "Age", "Sex", "ALB", "ALP", "ALT", "AST", "BIL",
        "CHE", "CHOL", "CREA", "GGT", "PROT",
    ]
    for col in feature_cols:
        if col != "Sex":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    X = df[feature_cols]
    y = df["target"]
    numeric = [c for c in feature_cols if c != "Sex"]
    return X, y, numeric


def main():
    print("=== HEART ===")
    X, y, num = prepare_heart()
    heart = _fit_bundle(
        X,
        y,
        num,
        RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        use_smote=True,
        calibrate=True,
        threshold=0.45,
    )
    joblib.dump(heart, f"{OUT_DIR}/heart_model.pkl")

    print("\n=== STROKE ===")
    X, y, num = prepare_stroke()
    # Heavy imbalance: SMOTE + scale_pos_weight-style depth control
    stroke = _fit_bundle(
        X,
        y,
        num,
        XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            reg_lambda=1.2,
            eval_metric="logloss",
            random_state=42,
        ),
        use_smote=True,
        smote_ratio=0.45,
        calibrate=False,
        threshold=0.28,
    )
    joblib.dump(stroke, f"{OUT_DIR}/stroke_model.pkl")

    print("\n=== HEPATITIS ===")
    X, y, num = prepare_hepatitis()
    hep = _fit_bundle(
        X,
        y,
        num,
        XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            objective="binary:logistic",
            random_state=42,
        ),
        use_smote=True,
        calibrate=True,
        threshold=0.45,
    )
    joblib.dump(hep, f"{OUT_DIR}/hepatitis_model.pkl")

    print(f"\nSaved models to {OUT_DIR}/")


if __name__ == "__main__":
    main()

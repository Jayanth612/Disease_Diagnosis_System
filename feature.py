import pandas as pd

HEART_FEATURES = [
    "Age", "Sex", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak",
    "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA",
    "RestingECG_Normal", "RestingECG_ST", "ExerciseAngina_Y",
    "ST_Slope_Flat", "ST_Slope_Up",
]

HEPATITIS_FEATURES = [
    "Age", "Sex", "ALB", "ALP", "ALT", "AST", "BIL",
    "CHE", "CHOL", "CREA", "GGT", "PROT",
]

STROKE_FEATURES = [
    "age", "hypertension", "heart_disease", "ever_married",
    "avg_glucose_level", "bmi", "gender_Male", "gender_Other",
    "work_type_Govt_job", "work_type_Never_worked", "work_type_Private",
    "work_type_Self-employed", "work_type_children",
    "Residence_type_Rural", "Residence_type_Urban",
    "smoking_status_Unknown", "smoking_status_formerly smoked",
    "smoking_status_never smoked", "smoking_status_smokes",
]


def _frame(data: dict, columns: list) -> pd.DataFrame:
    row = {col: data.get(col, 0) for col in columns}
    df = pd.DataFrame([row], columns=columns)
    return df.apply(pd.to_numeric, errors="coerce").fillna(0)


def get_features_for_heart(data: dict):
    return _frame(data, HEART_FEATURES)


def get_features_for_hepatitis(data: dict):
    return _frame(data, HEPATITIS_FEATURES)


def get_features_for_stroke(data: dict):
    return _frame(data, STROKE_FEATURES)


def prepare_features(disease_type: str, input_data: dict):
    key = disease_type.lower().strip()
    if key == "heart":
        return get_features_for_heart(input_data)
    if key == "hepatitis":
        return get_features_for_hepatitis(input_data)
    if key == "stroke":
        return get_features_for_stroke(input_data)
    raise ValueError("Invalid disease type. Choose from 'heart', 'hepatitis', 'stroke'.")

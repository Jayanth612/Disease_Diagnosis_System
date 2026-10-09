from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from feature import prepare_features
import joblib
import os

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": [
            "http://127.0.0.1:5000",
            "http://localhost:5000",
            "http://0.0.0.0:5102",
        ]
    }
})

MODELS = {}
for name in ["heart", "stroke", "hepatitis"]:
    path = f"Trained_Models/{name}_model.pkl"
    if os.path.exists(path):
        MODELS[name] = joblib.load(path)
        print(f"Loaded {path}")
    else:
        print(f"Model not found: {path}")


@app.route("/")
def home():
    return send_from_directory("templates", "index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models": list(MODELS.keys()),
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=True) or {}
        disease = (data.get("disease") or "").lower().strip()
        features = data.get("features")

        if disease not in MODELS:
            return jsonify({"error": f"Invalid disease: {disease}"}), 400
        if not isinstance(features, dict):
            return jsonify({"error": "features must be an object"}), 400

        df = prepare_features(disease, features)
        model = MODELS[disease]

        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(df)[0][1])
            threshold = getattr(model, "threshold", 0.5)
            pred = int(proba >= threshold)
        else:
            pred = int(model.predict(df)[0])

        risk = "high" if pred == 1 else "low"
        prob_txt = f"{proba:.1%}" if proba is not None else "n/a"
        if pred == 1:
            label = f"High Risk — estimated probability {prob_txt}"
        else:
            label = f"Low Risk — estimated probability {prob_txt}"

        return jsonify({
            "prediction_text": label,
            "raw": {
                "prediction": pred,
                "probability": round(proba, 4) if proba is not None else None,
                "risk": risk,
            },
            "input_preview": df.to_dict(orient="records"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

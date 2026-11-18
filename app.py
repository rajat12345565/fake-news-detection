from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

MODEL_PATH = Path("model.joblib")
app = Flask(__name__)
model = None


def load_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("Model not found. Add your dataset and run 'python train.py' first.")
        model = joblib.load(MODEL_PATH)
    return model


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    text = payload.get("text", "") if payload is not None else request.form.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Please enter article text."}), 400
    if len(text) > 50_000:
        return jsonify({"error": "Article text must be 50,000 characters or fewer."}), 400

    try:
        predictor = load_model()
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503

    prediction = predictor.predict([text])[0]
    confidence = float(max(predictor.predict_proba([text])[0]))
    return jsonify({"prediction": prediction, "confidence": round(confidence * 100, 2)})


if __name__ == "__main__":
    app.run(debug=False)

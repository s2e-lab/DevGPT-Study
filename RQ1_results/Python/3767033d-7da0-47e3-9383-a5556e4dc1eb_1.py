from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("confidence_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]
    prediction = model.predict([text])[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)

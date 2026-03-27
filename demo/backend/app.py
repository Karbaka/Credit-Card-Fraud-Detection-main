from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import json
import numpy as np
import os
import random

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

BASE = os.path.dirname(__file__)

with open(os.path.join(BASE, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE, 'features.json')) as f:
    features = json.load(f)

with open(os.path.join(BASE, 'samples.json')) as f:
    samples = json.load(f)


@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        values = [float(data.get(feat, 0)) for feat in features]
        arr = np.array(values).reshape(1, -1)
        prediction = int(model.predict(arr)[0])
        proba = model.predict_proba(arr)[0]
        fraud_prob = round(float(proba[1]) * 100, 2)
        return jsonify({
            'prediction': prediction,
            'fraud_probability': fraud_prob,
            'label': 'FRAUD' if prediction == 1 else 'LEGITIMATE'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/sample', methods=['GET'])
def sample():
    kind = request.args.get('type', 'normal')
    pool = samples.get(kind, samples['normal'])
    row = random.choice(pool)
    return jsonify(row)


if __name__ == '__main__':
    app.run(debug=True, port=5000)

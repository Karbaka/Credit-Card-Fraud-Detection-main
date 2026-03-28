from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import json
import numpy as np
import os
import random

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))

# Load samples
with open(os.path.join(BASE, 'samples.json')) as f:
    samples = json.load(f)

with open(os.path.join(BASE, 'features.json')) as f:
    features = json.load(f)

# Load or retrain model
def load_model():
    pkl_path = os.path.join(BASE, 'model.pkl')
    try:
        with open(pkl_path, 'rb') as f:
            m = pickle.load(f)
        print("Model loaded from pkl")
        return m
    except Exception as e:
        print(f"pkl load failed ({e}), retraining...")
        return retrain_model(pkl_path)

def retrain_model(pkl_path):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Build training data from samples
    normal = samples['normal']
    fraud  = samples['fraud']
    all_samples = normal + fraud

    X = [[s[f] for f in features] for s in all_samples]
    y = [s['Class'] for s in all_samples]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestClassifier(criterion='entropy', n_estimators=50, random_state=42)
    m.fit(X_train, y_train)

    with open(pkl_path, 'wb') as f:
        pickle.dump(m, f)
    print("Model retrained and saved")
    return m

model = load_model()


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

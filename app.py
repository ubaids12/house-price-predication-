
import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'decision_tree_model.pkl')
print("Loading model from:", MODEL_PATH)

model = joblib.load(MODEL_PATH)


FEATURE_ORDER = [
    'number of bedrooms',
    'number of bathrooms',
    'living area',
    'lot area',
    'number of floors',
    'waterfront present',
    'number of views',
    'living_area_renov',
    'lot_area_renov',
    'Distance from the airport',
    'Lattitude',
    'Longitude'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = []
        for field in FEATURE_ORDER:
            if field not in request.form:
                return jsonify({'error': f'Missing form field: {field}'}), 400
            try:
                values.append(float(request.form[field]))
            except ValueError:
                return jsonify({'error': f'Invalid numeric value for field: {field}'}), 400

        features = np.array([values])
        pred = model.predict(features)[0]
        return jsonify({'price': round(float(pred), 2)})

    except Exception as e:
       
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
   
    app.run(debug=True)


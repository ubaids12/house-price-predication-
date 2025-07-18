🏠 House Price Prediction

An end-to-end Machine Learning + Web App project that predicts residential property prices from user-supplied characteristics (e.g., square footage, bedrooms, bathrooms, location). The repository demonstrates the full lifecycle of an ML solution: data preparation, model development, evaluation, persistence, and an interactive web front end for real-time inference.

📘 Table of Contents

Introduction

Project Goals

What This Project Does

Dataset Overview

Features Used in the Model

Project Workflow (Step-by-Step)

Project Structure

Installation

Quick Start: Run the App

Using the Web App (How To Use Properly)

Re-train / Update the Model

Software & Libraries

Configuration & Environment Variables

Testing

Troubleshooting

Future Improvements

Contributing

License

Contact

Introduction

Accurately estimating property prices helps buyers, sellers, realtors, and investors make informed decisions. This project walks through building a Decision Tree Regression model to predict house prices and deploys it behind a Flask-powered web interface so anyone can input property details and receive an instant estimate.

The repository is designed for learning and demonstration: if you're new to Machine Learning deployment, you can see how raw CSV data transforms into an interactive application.

Project Goals

Primary Goals

Build a supervised regression model to estimate house prices.

Demonstrate data preprocessing best practices (missing values, encoding, scaling).

Provide a simple, production-style workflow for saving/loading trained models.

Expose the model via a lightweight Flask backend and HTML form UI.

Learning Goals

Understand end-to-end ML pipelines.

Practice separating experimentation (Jupyter Notebook) from application code (Flask app).

Explore how to validate a model and monitor performance when updating data.

What This Project Does

At a high level, the project:

Loads and cleans historical property data.

Engineers and selects predictive features.

Trains a Decision Tree Regressor.

Evaluates model performance (R², MAE, RMSE).

Serializes (“pickles”) the trained model for reuse.

Serves a web form where users enter property attributes.

Returns a predicted price in real time.

Dataset Overview

Note: Replace the placeholder details below with specifics about your dataset once finalized.

Column

Description

Example

square_feet

Total finished living area

1850

bedrooms

Total number of bedrooms

3

bathrooms

Total number of baths (full + partial weighted)

2.5

location

Neighborhood / city / zip grouping

"Downtown"

year_built

Year the home was constructed

2005

lot_size

Lot area (sq ft or acres)

7405

garage_spaces

Number of garage stalls

2

price

Target – historical sale price

325000

If your dataset contains additional fields (quality ratings, condition scores, distance to city center, etc.), document them here.

Features Used in the Model

The baseline model uses a subset of the most predictive and widely available features. Default set:

square_feet

bedrooms

bathrooms

location (categorical → encoded)

year_built

Optional/extended features (if data available):

lot_size

garage_spaces

Quality scores (e.g., overall condition)

Proximity metrics (schools, transport)

You can enable/disable features in the training notebook or pipeline script.

Project Workflow (Step-by-Step)

Below is the recommended path if you're cloning this repo to learn or extend it.

1. Explore the Data

Open model_training.ipynb and load the dataset from data/. Inspect column types, ranges, and missing values.

2. Clean & Prepare

Drop or impute missing values.

Encode categorical variables (OneHot or Ordinal).

Optionally scale numeric features (Decision Trees do not require scaling but downstream models might).

3. Split Data

Train/test split (e.g., 80/20). Optionally add validation split or cross-validation.

4. Train Model

Use DecisionTreeRegressor from scikit-learn. Tune hyperparameters such as max_depth, min_samples_split, min_samples_leaf, and max_features.

5. Evaluate

Compute:

R² Score – variance explained

MAE – average absolute error

RMSE – penalizes large errors
Compare train vs. test to check for overfitting.

6. Save Model

Serialize trained model to model/house_price_model.pkl (default path) using joblib or pickle.

7. Connect to Web App

app.py loads the trained model at startup. The HTML form (in templates/) collects user inputs, which are transformed and passed into the model for prediction.

8. Deploy / Run Locally

Run Flask locally (development mode) or deploy on a platform like Render, Railway, or Heroku.

Project Structure

House-Price-Prediction/
│
├── data/                      # Raw & processed datasets
│   └── housing_data.csv       # Example dataset (not tracked if large)
│
├── model/                     # Saved model(s) and encoders
│   ├── house_price_model.pkl
│   └── encoder.pkl            # For categorical features (if used)
│
├── notebooks/
│   └── model_training.ipynb   # ML experimentation notebook
│
├── templates/                 # HTML templates for Flask
│   └── index.html             # Form UI
│
├── static/                    # CSS / JS / images
│   └── style.css
│
├── app.py                     # Flask application entry point
├── inference.py               # Helper functions: load model, preprocess, predict
├── preprocess.py              # Reusable preprocessing steps (optional)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Ignore data/model artifacts as needed
└── README.md                  # This file

Installation

Tested with Python 3.9+. Earlier versions may work but are not guaranteed.

Clone the repository

git clone https://github.com/your-username/House-Price-Prediction.git
cd House-Price-Prediction

Create & activate a virtual environment (recommended)

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

Quick Start: Run the App

python app.py

The development server will start (default):

http://127.0.0.1:5000

Open that URL in your browser.

If you see a form asking for square footage, bedrooms, etc.—you’re good to go.

Using the Web App (How To Use Properly)

Open the app in your browser.

Enter values for each field:

Square Footage: Use whole numbers (e.g., 1800).

Bedrooms / Bathrooms: Use numeric counts; decimals allowed for half baths.

Location: Choose from dropdown (must match values used during training).

Any extra fields will be listed if enabled.

Click Predict Price.

The app returns an estimated price (USD by default—adjust as needed in config).

⚠️ Important: Predictions are based on historical data and model assumptions. Use results for educational/demo purposes—not financial decisions—unless validated with real market data.

Re-train / Update the Model

If you add new data or features, re-train:

Place your updated dataset in data/.

Open notebooks/model_training.ipynb.

Update the data path & feature list.

Re-run all cells to clean data, train, and evaluate.

Save the new model & encoder objects to model/.

Restart the Flask app so it picks up the updated files.

Command-line option (advanced): If you create train.py, you can retrain via:

python train.py --data data/housing_data.csv --model-path model/house_price_model.pkl

Software & Libraries

Core:

Python >= 3.9

Flask

Scikit-learn

Pandas

NumPy

Optional / Recommended:

Joblib (model serialization)

Matplotlib (EDA plots)

Jupyter / ipykernel (notebooks)

python-dotenv (env config)

Example requirements.txt:

flask
pandas
numpy
scikit-learn
joblib
matplotlib
python-dotenv

Add versions if you need reproducibility.

Configuration & Environment Variables

Create a .env file (optional) to customize runtime behavior:

FLASK_ENV=development
MODEL_PATH=model/house_price_model.pkl
ENCODER_PATH=model/encoder.pkl
HOST=0.0.0.0
PORT=5000
CURRENCY=USD

Your app.py can load these values using python-dotenv.

Testing

Basic tests can be added under tests/:

Unit tests: Ensure preprocessing handles expected types.

Smoke test: Load model & make a sample prediction.

Flask route test: POST form data → receive JSON prediction.

Example pytest command:

pytest -q

Troubleshooting

Issue

Possible Cause

Fix

App startup error: model not found

Wrong path

Check MODEL_PATH in .env

Predictions always same value

Model trained on 1 row or constant target

Re-check training data

Invalid input (location)

UI value not seen in training

Update encoder + retrain

NaN prediction

Missing or non-numeric input

Add input validation in app.py

Future Improvements

Support multiple ML models (Random Forest, XGBoost) & compare.

Add confidence intervals / prediction uncertainty.

Visualize feature importance in UI.

Upload CSV batch predictions.

Containerize with Docker.

Deploy to cloud (Render / AWS / Azure / GCP).

Contributing

Contributions welcome! Please:

Fork the repo

Create a feature branch

Commit changes with clear messages

Open a pull request

License

This project is licensed under the MIT License. See LICENSE for details.

Contact

Created by [ubaid]

LinkedIn: https://www.linkedin.com/in/ubaid ashraf

Email: ubaidashraf71@gmail.com

If you find this project helpful, please ⭐ the repository!

Enjoy building and learning!


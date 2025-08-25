from flask import Flask
from flask import render_template
from flask import request
import pickle
import pandas as pd
import numpy as np
# render_template() is used to render (generate) an HTML file (usually with Jinja2 templating) and send it back as an HTTP response to the browser.

app = Flask(__name__)#Creates the app object. 


# Use absolute or raw string paths to avoid escape sequence issues
import os

# Path to the model file
MODEL_PATH = r'D:\ML Projects\CarPricePredictor\LinearRegressionModel.pkl'
CSV_PATH = os.path.join(os.path.dirname(__file__), 'Cleaned Car.csv')


# Load model safely and print debug info
try:
    print("Loading model from:", MODEL_PATH)
    model = pickle.load(open(MODEL_PATH, 'rb'))
    print("Model type:", type(model))
    if hasattr(model, 'steps'):
        print("Pipeline steps:", model.steps)
except FileNotFoundError:
    model = None
    print(f"ERROR: Model file not found at {MODEL_PATH}. Please ensure 'LinearRegressionModel.pkl' exists.")

# Load CSV safely
try:
    car = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    car = None
    print(f"ERROR: CSV file not found at {CSV_PATH}. Please ensure 'Cleaned Car.csv' exists.")

@app.route('/')

def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    companies.insert(0, "Select Company")

    # Build the mapping: company -> list of models
    company_models = {}
    for company in companies:
        models = car[car['company'] == company]['name'].unique()
        company_models[company] = list(models)

    return render_template(
        'index.html',
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_type=fuel_type,
        company_models=company_models
    )

@app.route('/predict', methods = ['POST'])
def predict():

    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))
    
    prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]], columns = ['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    
    return str(np.round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug = True)


import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import xgboost as xgb
from flask_restful import reqparse
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))     # set file directory path
MODEL_PATH = os.path.join(APP_ROOT, "./models/gar_model.bin")  # set path to the model
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

@app.route('/submit', methods=['GET', 'POST'])  # submit the form
def make_prediction():
    features = [float(x) for x in request.form.values()] # take the values from the form as a list
    final_features = [np.array(features)]       # convert the values into a numpy array
    final_features = pd.DataFrame(final_features)
    final_features.columns = ['average_temp', 'out_count', 'population', 'num_household', 'visit', 'deliver_takeout']
    prediction = model.predict(final_features)      # pass the array into the model for prediction
    return render_template('prediction.html', prediction = prediction[0])  # render the prediction page

@app.route('/data')
def show_link():
    return render_template('data.html')

@app.route('/introduce')
def show_introduce():
    return render_template('introduce.html')


if __name__ == '__main__':
    app.run(debug=True)
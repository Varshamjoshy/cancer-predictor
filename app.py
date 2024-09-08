import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
  input_features = [float(x) for x in request.form.values()]
  features_value = [np.array(input_features)]

  features_name = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean',
       'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

  df = pd.DataFrame(features_value, columns=features_name)

  # Apply the scaler transformation
  scaled_features = scaler.transform(df)

  # Make prediction
  output = model.predict(scaled_features)

  if int(output) == 1:
      res_val = "malignant"
  else:
      res_val = "benign"


  return render_template('index.html', prediction_text='The tumor is {}'.format(res_val))

if __name__ == "__main__":
  app.run()

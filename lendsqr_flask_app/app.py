from flask import Flask, request, jsonify, send_file
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the model and encoder
model = joblib.load('model/loan_score_model.pkl')
encoder = joblib.load('model/label_encoder.pkl')

# Preprocessing function
def preprocess_data(df):
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = encoder.fit_transform(df[column].astype(str))
    return df

@app.route('/')
def home():
    return "Loan Score Prediction Model API"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Preprocess the data
        df_processed = preprocess_data(df)
        
        # Make predictions
        predictions = model.predict(df_processed)
        prediction_labels = ['Paid' if x == 2 else 'Unpaid' for x in predictions]
        
        # Save the results to a new DataFrame
        result_df = df.copy()
        result_df['Predicted Status'] = prediction_labels
        
        # Save the results to a CSV file
        output_file = 'predictions.csv'
        result_df.to_csv(output_file, index=False)
        
        return send_file(output_file, as_attachment=True, attachment_filename=output_file)

if __name__ == '__main__':
    app.run(debug=True)

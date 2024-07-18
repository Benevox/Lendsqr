from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('model/loan_score_model.pkl')

# Preprocessing function
def preprocess_data(df):
    le = joblib.load('model/label_encoder.pkl')
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = le.fit_transform(df[column].astype(str))
    return df

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)
    df = preprocess_data(df)
    predictions = model.predict(df)
    prediction_labels = ['Paid' if x == 2 else 'Unpaid' for x in predictions]
    return jsonify(prediction_labels)

if __name__ == '__main__':
    app.run(debug=True)

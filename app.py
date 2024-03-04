import sys
from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
from src.logger import logging

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'GET':
            return render_template('index.html')
        else:
            data = CustomData(
                ph=float(request.form.get('ph')),
                Hardness=float(request.form.get('Hardness')),
                Solids=float(request.form.get('Solids')),
                Chloramines=float(request.form.get('Chloramines')),
                Sulfate=float(request.form.get('Sulfate')),
                Conductivity=float(request.form.get('Conductivity')),
                Organic_carbon=float(request.form.get('Organic_carbon')),
                Trihalomethanes=float(request.form.get('Trihalomethanes')),
                Turbidity=float(request.form.get('Turbidity'))
            )
            pred_df = data.get_data_as_data_frame()
            logging.info("Before Prediction")

            predict_pipeline = PredictPipeline()
            logging.info("Mid Prediction")
            results = predict_pipeline.predict(pred_df)
            result_value = ""
            if results[0] == 1.0:
                result_value = "Safe for Human Consumption"
            else:
                result_value = "Not Safe for Human Consumption"
            return render_template('index.html', results=result_value)
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    url = 'http://127.0.0.1:5000'  # Default URL
    print("Flask web application is running. Visit", url)
    app.run(debug=True, use_reloader=True)

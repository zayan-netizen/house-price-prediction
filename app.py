import os
from flask import Flask, render_template, request
from src import *
from src.predict import predict_price, columns
app = Flask(__name__)

@app.route("/")
def home():

    locations = [col for col in columns if col not in ["total_sqft", "bath", "balcony"]]

    return render_template("index.html", locations=locations)

@app.route("/predict", methods=["POST"])
def predict():
    try:    
        sqft = float(request.form["SqFt"])
        bathroom = int(request.form["Bathroom"])
        balcony = int(request.form["Balcony"])
        location = request.form["Location"]

        locations = [col for col in columns if col not in ["total_sqft", "bath", "balcony"]]
        
        prediction = predict_price(sqft, bathroom, balcony, location)

        return render_template("index.html", prediction=prediction, sqft=sqft, bathroom=bathroom, balcony=balcony, location=location, locations=locations)
    
    except Exception as e:
        print(e)
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)

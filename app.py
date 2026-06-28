from flask import Flask, render_template, request
from src import *
from src.predict import predict_price
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    print(request.form)
    
    sqft = request.form["SqFt"]
    bathroom = request.form["Bathroom"]
    balcony = request.form["Balcony"]

    prediction = predict_price(sqft, bathroom, balcony)

    return render_template("index.html", prediction=prediction, sqft=sqft, bathroom=bathroom, balcony=balcony)

if __name__ == "__main__":
    app.run(debug = True)


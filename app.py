from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    print(request.form)
    
    sqft = request.form["SqFt"]
    bhk = request.form["BHK"]
    year = request.form["Year"]

    print("SqFt:", sqft)
    print("BHK:", bhk)
    print("Year:", year)
    
    prediction = 147

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug = True)


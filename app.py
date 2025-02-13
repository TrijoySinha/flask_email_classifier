#Flask - Lightweight Web framework
#render_template - function used to render HTML templates; Flask uses Jinja2 as its templating engine
#request - module used to handle incoming HTTP requests
#jsonify - function used to return JSON responses from flask app; Auto converts python dictionary to JSON format
from flask import Flask, render_template, request, jsonify
import pickle

cv = pickle.load(open("models/cv.pkl", "rb"))
clf = pickle.load(open("models/clf.pkl", "rb"))

app = Flask(__name__)          #instance of the Flask class

#Home Route
#GET - Fetches webpage from the server and #POST - will show smthng on clicking the button
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form.get("content")
    tokenized_email = cv.transform([email])
    prediction = clf.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return render_template("index.html", prediction=prediction, email=email)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)         #debug=True for auto update on server
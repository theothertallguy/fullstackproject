# %%
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # read form data and do prediction here
    return "Prediction placeholder"

@app.route("/logout")
def logout():
    return "Logged out"
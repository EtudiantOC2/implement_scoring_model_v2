# Import des packages nécessaires
import pandas as pd
import pickle
from flask import Flask, request, jsonify,render_template

# Création de l'app
app = Flask(__name__)

# Petit test Hello World
@app.route("/")
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.run(debug = True)
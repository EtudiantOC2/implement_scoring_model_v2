# Import des packages nécessaires
import pandas as pd
import pickle
import numpy as np
from flask import Flask, request, jsonify,render_template
import json

# Création de l'app
app = Flask(__name__)

# Petit test Hello World
@app.route("/")
def hello():
    return "Hello World!"

# Sauvegarde du modèle
model = pickle.load(open('C:\\Users\\pauline_castoriadis\\Documents\\implement_scoring_model_v2\\model\\model.pkl','rb'))

# Chargement data
def loading_csv_data(path):
    '''
    Retourne le dataframe à partir du chemin qu'on lui désigne
    '''
    df = pd.read_csv(path)
    df.drop("Unnamed: 0", axis = 1, inplace = True)
    return df

# chargement des donnees
data = loading_csv_data(r"C:\Users\pauline_castoriadis\Documents\implement_scoring_model_v2\data\df_test.csv")

# calcul du score et envoi d'un dico avec les infos id clients et prédiction
@app.route('/api/<int:applicant_loan_id>')
def mon_api(applicant_loan_id):
    features = np.array(data.drop(['applicant_loan_id'],axis = 1))
    val = model.predict_proba(features)
    prediction = model.predict_proba(features)[:,1]
    output = round(prediction[0]*100,2)
    dico = {'id': applicant_loan_id,
            'prediction':output}
    return jsonify(dico)

if __name__ == "__main__":
    app.run(debug = True)
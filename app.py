# Import des packages nécessaires
import pandas as pd
import pickle
from flask import Flask, request, jsonify,render_template

from pathlib import Path
import os
ROOT_PATH = Path(os.getenv("ROOT_PATH", "."))
DATA_PATH = ROOT_PATH / "data"
MODEL_PATH = ROOT_PATH / "model"

# Création de l'app
app = Flask(__name__)

# Petit test Hello World
@app.route("/")
def hello():
    return "Hello World!"

# Sauvegarde du modèle
model = pickle.load(open(MODEL_PATH / 'model.pkl','rb'))

# Chargement data
def loading_csv_data(path):
    '''
    Retourne le dataframe à partir du chemin qu'on lui désigne
    '''
    df = pd.read_csv(path)
    df.drop("Unnamed: 0", axis = 1, inplace = True)
    return df

# chargement des donnees
data = pickle.load(open(DATA_PATH / 'df_test'))

# Chargement prédiction
def loading_predicted_data(df,id_col):
    '''
    Réalise la prédiction pour l'ensemble des clients contenus dans la base donnée
    '''
    selection_col =  list(df.loc[:, (df.columns != id_col)])
    data_selection = df[selection_col].values
    prediction =  pd.DataFrame(data = model.predict_proba(data_selection)[:, 0],columns=['prediction'])
    result = pd.concat([df, prediction], axis = 1)
    return result

predicted_data = loading_predicted_data(data,'applicant_loan_id')

# calcul du score et envoi d'un dico avec les infos id clients et prédiction
@app.route('/api/<int:applicant_loan_id>')
def mon_api(applicant_loan_id):
    id_data = predicted_data[predicted_data['applicant_loan_id'] == int(applicant_loan_id)] # Données
    output = round(id_data['prediction'].iat[0,],3) # Prédiction
    dico = {'id': applicant_loan_id,
            'prediction':output}
    return jsonify(dico)

if __name__ == "__main__":
    app.run(debug = True)
# Import des packages
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify,render_template
import json

# Chargement modèle
model = pickle.load(open('C:\\Users\\pauline_castoriadis\\Documents\\implement_scoring_model\\model\\best_model.pkl','rb'))

# Chargement data
def loading_csv_data(path):
    '''
    Retourne le dataframe à partir du chemin qu'on lui désigne
    '''
    df = pd.read_csv(path)
    df.drop("Unnamed: 0", axis = 1, inplace = True)
    return df
data = loading_csv_data(r"C:\Users\pauline_castoriadis\Documents\implement_scoring_model\data\df_test.csv")

# Création de l'application
app = Flask(__name__)
app.config["DEBUG"] = True

# Création de l'URL d'entrée
@app.route('/', methods = ['GET'])
def home():
    return render_template('index.html')

# URL Predict lorsque l'employé a rentré l'ID client
@app.route('/predict',methods = ['POST'])
def predict():
    # Prédiction
    id_client = [int(x) for x in request.form.values()]
    features = np.array(data[data['applicant_loan_id'] == id_client[0]].drop(['applicant_loan_id'],axis = 1))
    prediction = model.predict_proba(features)[:,1]
    output = round(prediction[0]*100,2)
    dico = {'id':id_client,
            'prediction':output}
    return jsonify(dico)
    
    # Lien avec HTML
    #if output <= 50:
        #return render_template('index.html', prediction_text = '{} chances de ne pas rembourser son prêt : le dossier du client pourra être ré-étudié'.format(output))
    #elif output >= 75:
        #return render_template('index.html', prediction_text = '{} chances de ne pas rembourser son prêt : le dossier doit être refusé'.format(output))
    #else:
        #return render_template('index.html', prediction_text = '{} chances de ne pas rembourser son prêt : le dossier doit être mis sous surveillance'.format(output))

if __name__ == '__main__' :
    app.run()
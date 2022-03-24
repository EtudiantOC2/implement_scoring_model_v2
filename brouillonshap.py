#------------- Configuration de l'environnement streamlit et des packages python -------------

# Configuration préalable de Streamlit via la commande "streamlit run dashboard.py" 

# Import des packages nécessaires
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import shap
from flask import Flask
import requests
from urllib.error import URLError

# Configuration largeur dashboard
st.set_page_config(layout = "wide")


#------------- Chargement des données nécessaires à la formalisation du dashboard -------------

# Chargement dataset
def loading_csv_data(path):
    '''
    Retourne le dataframe à partir du chemin désigné par l'utilisateur
    '''
    df = pd.read_csv(path)
    df.drop("Unnamed: 0", axis = 1, inplace = True)
    return df

# Chargement prédiction
def loading_predicted_data(df,id_col):
    '''
    Réalise la prédiction pour l'ensemble des clients contenus dans la base donnée
    '''
    selection_col =  list(df.loc[:, (df.columns != id_col)])
    data_selection = df[selection_col].values
    prediction =  pd.DataFrame(data = loaded_model.predict_proba(data_selection)[:, 0],columns=['prediction'])
    result = pd.concat([df, prediction], axis = 1)
    return result

# Chargement groupes clients
def defining_group(df):
    '''
    Rajouter une colonne de groupe selon la probabilité de remboursement
    '''
    if (df['prediction'] < 0.50):
        return "restudy"
    elif (df['prediction'] > 0.75):
        return "refuse"
    else:
        return "monitor"

# Dataset
data = loading_csv_data(r"C:\Users\pauline_castoriadis\Documents\implement_scoring_model\data\df_test.csv") # Dataset

# Meilleur modèle ML
loaded_model = pickle.load(open('C:\\Users\\pauline_castoriadis\\Documents\\implement_scoring_model\\model\\model.pkl','rb')) # Modèle ML

# Prédiction
predicted_data = loading_predicted_data(data,'applicant_loan_id')

# Groupes clients
predicted_data['group'] = predicted_data.apply(defining_group, axis = 1)

# Liste id clients
applicant_id_list = data['applicant_loan_id'].tolist()

# Choix id client
st.sidebar.write("# Choix du client")
selected_id = st.sidebar.selectbox("", applicant_id_list,index = 1)
st.sidebar.write("# ID sélectionné:  ", selected_id)

# Données relatives au client sélectionné
id_data = predicted_data[predicted_data['applicant_loan_id'] == int(selected_id)] # Données
prediction_id = round(id_data['prediction'].iat[0,],2) # Prédiction
group_id = id_data['group'].iat[0,] # Groupe

st.write(id_data.iloc[:, : 27])
#------------- Configuration de l'environnement streamlit et des packages python -------------

# Configuration préalable de Streamlit via la commande "streamlit run dashboard.py" 

# Import des packages nécessaires
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import shap
from flask import Flask
import requests
from urllib.error import URLError
from urllib.request import urlopen
import json

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
data = loading_csv_data(r"C:\Users\pauline_castoriadis\Documents\implement_scoring_model_v2\data\df_test.csv") # Dataset

# Meilleur modèle ML
loaded_model = pickle.load(open('C:\\Users\\pauline_castoriadis\\Documents\\implement_scoring_model_v2\\model\\model.pkl','rb')) # Modèle ML

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

# Données relatives au client sélectionné à partir de l'api correspondante
id_data = predicted_data[predicted_data['applicant_loan_id'] == int(selected_id)] # Données
id = str(selected_id)
url = "http://127.0.0.1:5000/api/%s" % id
response = urlopen(url)
data_json = json.loads(response.read())
prediction_id = float(data_json["prediction"])# Prédiction
group_id = id_data['group'].iat[0,] # Groupe


#------------- Formatting des titres et définition des pages dashboard utilisables -------------

# Format html des titres
def formatting_title_1(title):
    return st.markdown(f"<h1 style='text-align: center; color: white;font-size:30px'>{title}</h1>", unsafe_allow_html = True)

def formatting_title_2(title):
    return st.markdown(f"<h1 style='text-align: left; color: white;font-size:20px'>{title}</h1>", unsafe_allow_html = True)

def formatting_title_3(title1,title2,title3):
    return st.markdown(f"<h1 style='text-align: left; color: white;font-size:20px'><ul><li>{title1}</li><li>{title2}</li><li>{title3}</li></ul></h1>", unsafe_allow_html = True)

def formatting_title_4(title):
    return st.markdown(f"<h1 style='text-align:right; color: white;font-size:15px'>{title}</h1>", unsafe_allow_html = True)

def formatting_title_5(title):
    return st.markdown(f"<h1 style='text-align:left; color: white;font-size:15px'>{title}</h1>", unsafe_allow_html = True)
    
# Définition des différentes pages de notre dashboard    

st.sidebar.write("# Choix de la page")

page = st.sidebar.selectbox('Sélectionner la page correspondante',
  ['Introduction','Prédiction score','Analyse client','Rapport client'])

st.sidebar.info('**Si vous constatez un bug ou avez un besoin spécifique, contactez-vous!**')


#------------- Page 1 de notre dashboard : introduction du problème -------------

if page == 'Introduction':

    # Introduction sur les objectifs du dashboard

    formatting_title_1('Dashboard interactif à destination des gestionnaires de la relation client ')
    
    st.markdown("<hr/>",unsafe_allow_html = True)
    
    titre = 'Prêt à dépenser a développé pour vous un dashboard interactif qui permettra une explication transparente des décisions d’octroi de crédit et une présentation claire des informations personnelles de chacun de vos clients'
    formatting_title_2(titre)

    titre = 'Ce dashboard vous permettra de :'
    formatting_title_2(titre)
    
    titre1 = 'Visualiser le score et l’interprétation du score pour chaque client'
    titre2 = 'Visualiser des informations relatives à un client'
    titre3 = 'Comparer les informations relatives à un client à un groupe de client similaire'
    formatting_title_3(titre1,titre2,titre3)

    # Copyright & marque

    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("")

    logo = Image.open(r'C:\Users\pauline_castoriadis\Documents\implement_scoring_model_v2\images\logo.jpg')
    
    copyright_1,copyright_2 = st.columns(2)

    with copyright_1:
        formatting_title_4('Produit par')

    with copyright_2:
        st.image(logo,width = 70)

    st.markdown("<hr/>",unsafe_allow_html = True)

#------------- Page 2 de notre dashboard : prédiction du score par client  -------------

elif page == 'Prédiction score':
    
    formatting_title_1('Dashboard interactif à destination des gestionnaires de la relation client ')

    st.markdown("<hr/>",unsafe_allow_html = True)

    # Calcul & affichage de la probabilité de faire défaut
   
    def displaying_id_group(id):
        '''
        Permet d'afficher le groupe du client
        '''
        if id == "monitor":
            titre = 'A surveiller'
            formatting_title_2(titre)
        elif id == "restudy":
            titre = 'A étudier'
            formatting_title_2(titre)
        else:
            titre = 'A refuser'
            formatting_title_2(titre)

    def displaying_id_comment(id):
        '''
        Permet d'afficher un commentaire en fonction du groupe client
        '''
        if id == "monitor":
            titre = 'Le client sélectionné est dans la moyenne, mais nécessite une surveillance de la part de nos services pour cette demande de crédit et/ou des aménagements spécifiques'
            formatting_title_2(titre)
        elif id == "restudy":
            titre = 'Le client sélectionné a peu de chances de ne pas rembourser son prêt ; cela vaut le coup de retravailler et ré-étudier son dossier'
            formatting_title_2(titre)
        else:
            titre = 'Le client sélectionné a une forte probabilité de ne pas rembourser son prêt, il est possible de rendre une décision favorable à sa demande, uniquement dans des cas exceptionnels'
            formatting_title_2(titre)
   
    def displaying_id_comment_small(id):
        '''
        Permet d'afficher un commentaire en fonction du groupe client
        '''
        if id == "monitor":
            titre = 'Le client sélectionné est dans la moyenne, mais nécessite une surveillance de la part de nos services pour cette demande de crédit et/ou des aménagements spécifiques'
            formatting_title_5(titre)
        elif id == "restudy":
            titre = 'Le client sélectionné a peu de chances de ne pas rembourser son prêt ; cela vaut le coup de retravailler et ré-étudier son dossier'
            formatting_title_5(titre)
        else:
            titre = 'Le client sélectionné a une forte probabilité de ne pas rembourser son prêt, il est possible de rendre une décision favorable à sa demande, uniquement dans des cas exceptionnels'
            formatting_title_5(titre)

    def chart_prediction(prediction_value,titre,hauteur,largeur):
        fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction_value,
        domain = {'row': 0, 'column': 0},
        title = {'text': titre},
        gauge = {'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "grey"},
        'bar': {'color': "grey"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 0.5], 'color': '#228B22'},
            {'range': [0.5, 75], 'color': '#FF8C00'},
            {'range': [0.75, 1], 'color': '#B22222'}],
        'threshold': {
            'line': {'color': "red", 'width': 5},
            'thickness': 0.75,
            'value': 0.75}}))
        fig.update_layout(autosize=False,width = largeur,height = hauteur)
        st.plotly_chart(fig)

    id_prediction,id_description = st.columns([3,3])
    
    with id_prediction:
        chart_prediction(prediction_id,'Client sélectionné',300,500)

    with id_description:
        formatting_title_2('Le client sélectionné appartient donc au groupe suivant :')
        displaying_id_group(group_id)
    
    group1,group2,group3 = st.columns([2,2,2])

    with group1:
        group1_prediction_data = predicted_data[predicted_data['group'] == "restudy"]
        chart_prediction(float(group1_prediction_data.groupby('group')['prediction'].mean()),'Dossiers à ré-étudier',300,300)
        displaying_id_comment_small('restudy')

    with group2:
        group2_prediction_data = predicted_data[predicted_data['group'] == "monitor"]
        chart_prediction(float(group2_prediction_data.groupby('group')['prediction'].mean()),'Dossiers à surveiller',300,300)
        displaying_id_comment_small('monitor')

    with group3:
        group3_prediction_data = predicted_data[predicted_data['group'] == "refuse"]
        chart_prediction(float(group3_prediction_data.groupby('group')['prediction'].mean()),'Dossiers à refuser',300,300)
        displaying_id_comment_small('refuse')

    st.markdown("<hr/>",unsafe_allow_html = True)


#------------- Page 3 de notre dashboard : analyse de nos clients/groupes  -------------

elif page == 'Analyse client':

    formatting_title_1('Dashboard interactif à destination des gestionnaires de la relation client ')
    
    st.markdown("<hr/>",unsafe_allow_html = True)
    
    # Quelques indicateurs
    formatting_title_2('Découvrez quelques indicateurs clés à propos du client sélectionné')

    def displaying_numerical_kpi(titre,df,id,col):
        '''
        Affiche un KPI par colonne et par client sélectionné, pour les colonnes numériques
        '''
        st.markdown(titre)
        kpi = round(df.at[df[df['applicant_loan_id']==int(id)].index[0],col],0)
        st.markdown(f"<h1 style='text-align: center; color: red;'>{kpi}</h1>", unsafe_allow_html = True)

    def displaying_sex_kpi(titre,df,id,col):
        '''
        Affiche le sexe du client sélectionné
        '''
        st.markdown(titre)
        kpi_value = df.at[df[df['applicant_loan_id']==int(id)].index[0],col]
        if kpi_value == 0 :
            kpi = "Femme"
        elif kpi_value == 1:
            kpi = "Homme"
        else:
            kpi = "Autre"
        st.markdown(f"<h1 style='text-align: center; color: red;'>{kpi}</h1>", unsafe_allow_html = True)
    
    def displaying_status_kpi(titre,df,id,col):
            '''
            Affiche le sexe du client sélectionné
            '''
            st.markdown(titre)
            kpi_value = df.at[df[df['applicant_loan_id']==int(id)].index[0],col]
            if kpi_value == 0 :
                kpi = "Marié(e) civilement"
            elif kpi_value == 1:
                kpi = "Marié(e)"
            elif kpi_value == 2:
                kpi = "Séparé(e)"  
            elif kpi_value == 3:
                kpi = "Célibataire"    
            elif kpi_value == 4:
                kpi = "Statut inconnu"   
            else:
                kpi = "Veuf/Veuve"
            st.markdown(f"<h1 style='text-align: center; color: red;'>{kpi}</h1>", unsafe_allow_html = True)
    
    kpi_1, kpi_2, kpi_3,kpi_4 = st.columns(4)
    
    with kpi_1:
        displaying_sex_kpi("**Sexe**",data,selected_id,'applicant_gender')
        
    with kpi_2:
        displaying_numerical_kpi("**Age**",data,selected_id,'applicant_age')
        
    with kpi_3:
        displaying_status_kpi("**Statut marital**",data,selected_id,'applicant_family_status')
    
    with kpi_4:
        displaying_numerical_kpi("**Revenus**",data,selected_id,'applicant_total_income')
    
    st.markdown("<hr/>",unsafe_allow_html = True)

    # Affichage des variables les plus importantes pour le client sélectionné
    
    formatting_title_2('Comprenez les variables les plus importantes pour la prédiction du client sélectionné')
    
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)
    
    X = id_data.iloc[:, : 26]
    explainer = shap.TreeExplainer(loaded_model)
    shap_values = explainer.shap_values(X)
    st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:]))
    
    st.markdown("<hr/>",unsafe_allow_html = True)
    
    formatting_title_2('Comparez les indicateurs de votre client aux autres groupes')

    # Choix de variables dans nos graphiques
    col1 = data.loc[:,~data.columns.isin(['applicant_loan_id', 'credit_payment_type','applicant_gender','flag_car_owner_applicant','applicant_best_education','applicant_family_status','applicant_housing_type','applicant_occupation'])].columns.tolist()
    var1 = st.sidebar.selectbox('Sélectionnez une première variable',(col1))

    col2 = data.loc[:,~data.columns.isin(['applicant_loan_id', 'credit_payment_type','applicant_gender','flag_car_owner_applicant','applicant_best_education','applicant_family_status','applicant_housing_type','applicant_occupation',var1])].columns.tolist()
    var2 = st.sidebar.selectbox('Sélectionnez une deuxième variable',(col2))
    
    # Data pour graphiques
    def data_for_chart(group,var):
        grouper = predicted_data.groupby('group')
        sub_df = pd.concat([pd.Series(v[var].tolist(), name = k) for k, v in grouper], axis = 1)
        sub_df = sub_df[[group]].dropna()
        sub_df = sub_df.rename(columns = {group: predicted_data[var].name})
        return sub_df
    
    # Data par groupe et selon les variables sélectionnées
    sub_df1 = pd.concat([data_for_chart('restudy',var1),data_for_chart('restudy',var2)], axis = 1).sample(n = 1000)
    sub_df2 = pd.concat([data_for_chart('monitor',var1),data_for_chart('monitor',var2)], axis = 1).sample(n = 1000)
    sub_df3 = pd.concat([data_for_chart('refuse',var1),data_for_chart('refuse',var2)], axis = 1).sample(n = 1000)

    # Graphiques à afficher
    def graph1(val,reference,titre):
        fig1 = go.Figure(go.Indicator(mode = "number+delta",value = val,delta = {"reference": reference, "valueformat": ".0f"},title = {"text": titre}))
        fig1.update_layout(width = 250,height = 400)
        st.plotly_chart(fig1)
    
    def graph2(df,col1,col2):
        fig2 = px.scatter(df, x=col1, y=col2)
        fig2.update_layout(width = 1000,height = 400)
        st.plotly_chart(fig2)
    
    formatting_title_5('Chaque graphique vous permet de comparer deux variables, ainsi que les performances du groupe en moyenne sur la première variable sélectionnée')
    
    chart_graph_1,chart_graph_2 = st.columns([1,4])

    with chart_graph_1:
        graph1(id_data.iloc[0][var1],sub_df1[var1].mean(),'Dossiers à ré-étudier')
   
    with chart_graph_2:
        graph2(sub_df1,var1,var2)

    chart_graph_3,chart_graph_4 = st.columns([1,4])

    with chart_graph_3:
        graph1(id_data.iloc[0][var1],sub_df2[var1].mean(),'Dossiers à surveiller')
   
    with chart_graph_4:
        graph2(sub_df2,var1,var2)

    chart_graph_5,chart_graph_6 = st.columns([1,4])

    with chart_graph_5:
        graph1(id_data.iloc[0][var1],sub_df3[var1].mean(),'Dossiers à refuser')
   
    with chart_graph_6:
        graph2(sub_df3,var1,var2)
   
    st.markdown("<hr/>",unsafe_allow_html = True)

#------------- Page 4 de notre dashboard : production de rapports clients par nos conseillers  -------------

elif page == 'Rapport client':

    formatting_title_2('Rédigez un rapport pour le client dont vous venez de regarder le dossier')
    
    def loading_excel_data(path):
        '''
        Retourne le dataframe à partir du chemin qu'on lui désigne
        '''
        df = pd.read_excel(path)
        return df
    
    report_data = loading_excel_data(r"C:\Users\pauline_castoriadis\Documents\implement_scoring_model_v2\data\reports.xlsx")
    
    form = st.form(key="annotation")
    
    with form: # Formulaire à remplur
        cols = st.columns((1, 1))
        author = cols[0].text_input("Nom du conseiller :")
        report_type = cols[1].selectbox(
        "Objet du rapport :", ["Dossier à ré-étudier", "Rendez-vous à fixer", "Documents manquants","Autre"], index=2)
        comment = st.text_area("Commentaire :")
        cols = st.columns(2)
        date = cols[0].date_input("Date de rapport :")
        report_severity = cols[1].slider("Criticité :", 1, 5, 2)
        submitted = st.form_submit_button(label = "Soumettre")
    
    if submitted:
        report = {'Client':selected_id,'Conseiller': author, 'Objet': report_type, 'Urgence': report_severity, 'Commentaire': comment,'date': str(date)}
        report_data = report_data.append(report, ignore_index = True)
        st.success("Merci, votre rapport a bien été enregistré")
        st.balloons()
    
    expander = st.expander("Voir tous les rapports produits")
    
    with expander:
        st.write(report_data)
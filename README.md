# implement_scoring_model_v2

OpenClassrooms - Projet 7 - Implementez un modèle de scoring

La société financière "Prêt à dépenser" propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.

L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.

Prêt à dépenser décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement. 

Le dashboard est disponible ici :https://prediction-client.herokuapp.com/ ; celui-ci permet de :
- Visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science
- Visualiser des informations descriptives relatives à un client
- Comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires

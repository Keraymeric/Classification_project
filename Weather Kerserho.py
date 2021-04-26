# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:22:43 2020

@author: keray
"""
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("weather.csv")


print("Nombres de lignes et type de l'objet")    
data.info() #renvoie les colonnes et le type de l'objet
data.hist() # affiche pour chaques colonnes un histogramme des donnnées


print("Visualisation des valeurs manquantes pour toutes les données")
print(data.isna()) # permet de voir les valeurs manquantes dans chaques colonnes

print("Somme des valeurs manquantes par colonnes")
print(data.isna().sum()) # fait la somme des valeurs manquantes de chaques colonnes

del data['Evaporation'] #supprime la colonne data
del data['Sunshine']
del data['Cloud9am']
del data['Cloud3pm']

for k in data['MaxTemp']: # Suppression des valeurs aberrantes de la température max
    if k>50:
        to_del = data[data["MaxTemp"] == k].index.tolist()
        data= data.drop(to_del)
        
data["WindGustDir"].fillna(method = 'bfill', inplace = True) # remplace automatiquement(inplace=true) les NaN par la valeur correcte suivante
data["WindDir9am"].fillna(method = 'bfill', inplace = True) # 
data["WindDir3pm"].fillna(method = 'bfill', inplace = True)
data["RainToday"].fillna(method = 'bfill', inplace = True)
print("Vérification du remplacement des données textuelles par la valeur correcte suivante")
print(data.isna().sum()) # vérification  


data.fillna(data.median(), inplace = True) # remplace automatiquement les valeurs manquantes par la valeur médiane
print("Vérification du remplacement des données numériques par la médiane")
print(data.isna().sum()) # vérification     

encoder = LabelEncoder() # création de l'objet de la classe LabelEncoder
for k in data: 
    if data[k].dtypes==object: # recherche des colonnes de type objet ou de variable de type text
        encoder.fit(data[k]) # recense les valeurs de type textes et leur attribue  une valeur correspondante 
        data[k]=encoder.transform(data[k]) # Remplace les valeurs des colonnes par le tableau de numpy crée précedemment 

choix=['RainToday','Humidity3pm'] # On garde seulement les deux variables les plus corrélées avec RainTomorrow

X=data[choix] # Isolation des variables d'entrées Humidity3pm et RainToday
y=data['RainTomorrow'] # Isolation de la variable de sortie RainTomorrow

X=X.values
y=y.values

scaler = StandardScaler().ﬁt(X) # normalisation des valeurs ( moyenne à 0 et écart type de 1)
X[:] = scaler.transform(X) # remplacement des valeurs du dataframe par les valeurs normalisées en conservant le type DataFrame

for k in data:
    print("Calcul du coefficient de corrélation de la colonne ",k ," avec RainTomorrow")
    print(data['RainTomorrow'].corr(data[k])) # calcul des coefficients de corrélation pour chaques colonnes avec la variable de sortie "RainTomorrow"
# On sélectionne les colonnes RISK_MM RainToday et Humidity3pm

params=['Humidity3pm','RainToday','RainTomorrow'] # création de la liste des variables intéressantes
scatter_matrix(data[params], alpha=0.2, figsize=(12,10),diagonal='kde') #Trace la matrice des graphiques
plt.show() 


X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.2) # Création des jeux de données (20% de la taille du jeu initial) et d'apprentissages (80% de la taille du jeu initial)


from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression() #
logisticRegr.fit(X_train, y_train) #Entrainement du modèle sur le jeu d'apprentissage ( Calcul des coefficients )

y_pred = logisticRegr.predict(X_test)  # Prédictions sur les données d'entrées du jeu de test


for k in range (19):
    print(y_test[k],y_pred[k]) #affichage des 20 premiers termes de la classe prédite et de la classe réelle

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score
# évalution quantitative du modèle
print("accuracy_score:", accuracy_score(y_test, y_pred))

print("confusion_matrix :", confusion_matrix(y_test,y_pred))

print("precision_score :", precision_score(y_test,y_pred))

print("f1_score :", f1_score(y_test, y_pred))

print("recall_score :", recall_score(y_test, y_pred))

from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=10)
for train, test in kf.split(X):
    print( "TRAIN:",train,"TEST:", test)

print(cross_val_score(logisticRegr,X,y,cv=10))
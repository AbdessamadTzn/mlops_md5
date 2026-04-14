import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train():

    print("Chargement des données...")
    df = pd.read_csv('data/customer_churn.csv')
    

    X = df.drop(['customer_id', 'churn'], axis=1)
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    print("Entraînement du modèle (Random Forest)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
 
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\nRésultats de l'évaluation :")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))
   
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(X.columns.tolist(), 'models/model_columns.pkl')
    
    print("\nModèle sauvegardé dans 'models/model.pkl'")

if __name__ == "__main__":
    train()

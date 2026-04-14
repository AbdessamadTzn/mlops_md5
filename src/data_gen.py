import pandas as pd
import numpy as np
import os

def generate_professional_data(n_samples=1000):
    np.random.seed(42)
    
    # Features de base
    customer_ids = [f"CUST-{i:04d}" for i in range(n_samples)]
    tenure = np.random.randint(1, 72, n_samples)  # Mois d'ancienneté
    monthly_charges = np.random.uniform(20, 120, n_samples)
    total_charges = tenure * monthly_charges + np.random.normal(0, 50, n_samples)
    
    # Comportement
    support_calls = np.random.poisson(2, n_samples) # Moyenne de 2 appels
    usage_frequency = np.random.uniform(0, 1, n_samples) # Score d'utilisation
    
    # Logique métier pour le Churn (Probabilité corrélée)
    # Plus ils appellent le support et moins ils utilisent, plus ils partent
    logit = (
        -2.0 
        + 0.5 * support_calls 
        - 3.0 * usage_frequency 
        - 0.02 * tenure 
        + 0.01 * monthly_charges
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (np.random.rand(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'support_calls': support_calls,
        'usage_frequency': usage_frequency,
        'churn': churn
    })
    
    # Sauvegarde
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/customer_churn.csv', index=False)
    print(f"Dataset généré : data/customer_churn.csv ({n_samples} lignes)")
    print(f"Taux de churn : {df.churn.mean():.2%}")

if __name__ == "__main__":
    generate_professional_data()

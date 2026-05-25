import pandas as pd
import numpy as np

def get_catalyst_pcs(catalyst_name, csv_path):
    """
    Récupère les valeurs de Principal Components (PC1 à PC10) pour un catalyseur donné.
    """
    try:
        df = pd.read_csv(csv_path, sep=';')
    except FileNotFoundError:
        return {"error": "Fichier introuvable"}
    
    subset = df[df['Catalyst'] == catalyst_name]
    if subset.empty:
        return {"error": "Aucune donnée trouvée pour ce catalyseur."}
        
    pc_features = [f"PC{i}" for i in range(1, 11)]
    available_pcs = [col for col in pc_features if col in df.columns]
    
    return subset[available_pcs].iloc[0].to_dict()


def pcs_to_catalyst(pcs, csv_path):
    """
    Trouve le catalyseur dont les composantes principales (PC) sont les plus proches 
    des valeurs données en utilisant la distance euclidienne.
    
    :param pcs: dictionnaire (ex: {'PC1': 1.2, ...}) ou liste/array de valeurs.
    """
    try:
        df = pd.read_csv(csv_path, sep=';')
    except FileNotFoundError:
        return {"error": "Fichier introuvable"}
        
    pc_features = [f"PC{i}" for i in range(1, 11)]
    available_pcs = [col for col in pc_features if col in df.columns]
    
    # 1. Convertir l'entrée en vecteur numpy
    if isinstance(pcs, dict):
        target_vector = np.array([pcs.get(col, 0.0) for col in available_pcs])
    else:
        target_vector = np.array(pcs)
        
    # 2. Chercher le catalyseur avec la distance minimale (vectorisé)
    cat_vectors = df[available_pcs].values
    distances = np.linalg.norm(cat_vectors - target_vector, axis=1)
    min_idx = np.argmin(distances)
    
    best_cat = df.iloc[min_idx]['Catalyst']
    min_dist = distances[min_idx]
                
    return {"catalyst": best_cat, "distance": min_dist}
    

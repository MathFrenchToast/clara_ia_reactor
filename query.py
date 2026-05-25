import pandas as pd

# 1. Chargement des données
df = pd.read_csv('v2c_featurized_ml_ready.csv')

# 2. Filtrage avec la méthode .query() (Syntaxe moderne avec backticks pour les espaces)
requete = "`Reaction Temperature` <= 800 and (`CH4 Conversion` >= 95 or `CO2 Conversion` >= 94)"
df_filtre = df.query(requete)
columns_to_keep = ['Reaction Temperature', 'CH4 Conversion', 'CO2 Conversion']
final_display_df = df_filtre[columns_to_keep]

# 3. Sauvegarde du résultat
# df_filtre.to_csv('filtered_results.csv', index=False)
print(final_display_df)

# Sources de données

Les données ne sont pas dans le dépôt. Elles sont téléchargées par le code (`python -m vrai_prix.dvf` puis `python -m vrai_prix.stations`) et gardées dans `data/raw/`.

## Ventes : DVF géolocalisées

Demandes de valeurs foncières (DGFiP), mises en forme et géolocalisées par Etalab. Licence Ouverte 2.0.

- Page : https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres-geolocalisees
- Fichiers : `https://files.data.gouv.fr/geo-dvf/latest/csv/<année>/departements/<département>.csv.gz`
- Utilisé : Paris, Hauts-de-Seine, Seine-Saint-Denis, Val-de-Marne, de 2021 à 2025.
- Téléchargé en novembre.

Les coordonnées sont celles du centre de la parcelle. Le nettoyage (ventes simples d'un appartement, surface de 9 à 300 m², prix de 1 500 à 30 000 €/m²) est dans `src/vrai_prix/dvf.py`. Le nombre de ventes restantes à chaque étape est écrit dans `data/processed/dvf_appartements_rapport.csv`.

## Stations : Île-de-France Mobilités

Jeu de données « Emplacement des gares et stations du réseau ferré d'Île-de-France », récupéré par l'API du portail en GeoJSON (voir `STATIONS_URL` dans `src/vrai_prix/stations.py`).

- Portail : https://data.iledefrance-mobilites.fr
- Utilisé : stations de métro et de RER, avec leur nom, leurs lignes et leur date de mise en service.
- Téléchargé en novembre.

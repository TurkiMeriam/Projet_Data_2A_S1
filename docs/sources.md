# Sources de données
(voir le README, section « Gestion des fichiers de données »).

## 1. Valeurs foncières (DVF géolocalisées, Etalab)

- **Page du jeu de données** : <https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres-geolocalisees>
- **URL de téléchargement** : `https://files.data.gouv.fr/geo-dvf/latest/csv/<année>/departements/<dép>.csv.gz`
  (définie dans `src/vrai_prix/config.py`)
- **Périmètre retenu** : départements 75, 92, 93, 94 ; années 2021 à 2025
- **Contenu utilisé** : date, nature de mutation, valeur foncière, type de local, surface réelle bâtie,
  nombre de pièces, coordonnées de la parcelle
- **Licence** : Licence Ouverte (Etalab)

## 2. Gares et stations du réseau ferré d'Île-de-France (Île-de-France Mobilités)

- **Jeu de données** : « Emplacement des gares et stations IDF », un point par station et par ligne
  (voir aussi <https://www.data.gouv.fr/datasets/gares-et-stations-du-reseau-ferre-dile-de-france-par-ligne>)
- **URL de téléchargement** : export GeoJSON de l'API du portail data.iledefrance-mobilites.fr
  (constante `STATIONS_URL` dans `src/vrai_prix/stations.py`)
- **Contenu utilisé** : mode (métro, RER), ligne, nom de la station, date de mise en service, géométrie
- **Filtre appliqué** : modes `METRO` et `RER` ; une station n'est prise en compte qu'à partir de sa date
  de mise en service

## Sources envisagées mais non utilisées

Fréquentation des stations (IDFM, RATP), contours IRIS, Base permanente des équipements (INSEE),
OpenStreetMap, DVF+ (Cerema). Elles pourraient prolonger l'étude (pondérer les stations par leur
fréquentation, tenir compte des aménités) mais n'interviennent dans aucun résultat.

# Le Vrai Prix de la Ville

Combien la proximité d'une station de métro ou de RER ajoute-t-elle au prix au m² d'un appartement ? Étude sur Paris et la petite couronne (75, 92, 93, 94), à partir des ventes réelles publiées en open data entre 2021 et 2025.

Projet de programmation (ENSAE, 2A, S1) réalisé à trois : BAHAN Karafa Ashley Kevin, TURKI Meriam, NAFTI Mohamed Amin.

## Question de recherche

À commune, année et taille de logement comparables, un appartement proche d'une station se vend-il plus cher ? L'effet est-il le même à Paris et en petite couronne, et change-t-il d'une année à l'autre ?

L'analyse est descriptive : elle repose sur des statistiques simples (médianes, corrélations, comparaisons entre ventes comparables), sans modèle économétrique.

## Données

| Source | Contenu | Récupération |
|---|---|---|
| [DVF géolocalisées](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres-geolocalisees) (Etalab) | Ventes immobilières : prix, surface, pièces, type de bien, coordonnées de la parcelle | Téléchargement automatique, un fichier par année et par département (URL publique) |
| [Gares et stations du réseau ferré d'Île-de-France](https://data.iledefrance-mobilites.fr) (Île-de-France Mobilités) | Un point par station et par ligne : mode, ligne, date de mise en service | Export GeoJSON par l'API du portail|

### Gestion des fichiers de données

Les données brutes sont trop volumineuses pour Git (plus de 100 Mo pour le DVF des quatre départements). Elles ne sont donc pas dans le dépôt : le code les télécharge depuis leur URL publique et les garde en cache dans `data/` (Les données DVF (brutes et nettoyées) et les stations IDFM sont versionnées dans `data/` pour que les notebooks tournent sans téléchargement). Chaque étape ne refait son travail que si le fichier est absent, ce qui évite de retélécharger à chaque exécution.


## Installation

```bash
git clone https://github.com/TurkiMeriam/Projet_Data_2A_S1.git
cd Projet_Data_2A_S1
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # rend `import vrai_prix` disponible partout
```

## Exécution

Depuis la racine du dépôt :

```bash
# 1. Télécharge et nettoie le DVF (75, 92, 93, 94 ; 2021-2025)
python -m vrai_prix.dvf

# 2. Télécharge les stations IDFM et calcule la distance de chaque vente
python -m vrai_prix.stations
```

Les deux étapes ne retéléchargent pas ce qui est déjà présent. Quelques options qui nous seront utiles :

```bash
python -m vrai_prix.dvf --years 2024 2025 --deps 75      # sous-ensemble
python -m vrai_prix.dvf --skip-download                  # réutilise data/raw/
python -m vrai_prix.stations --modes METRO RER           # modes de transport retenus
python -m vrai_prix.stations --skip-download             # réutilise le GeoJSON déjà téléchargé
```

Fichiers produits (non versionnés) :

- `data/raw/dvf/<année>/<département>.csv.gz` : fichiers DVF bruts
- `data/raw/stations/emplacement_gares_idfm.geojson` : stations IDFM
- `data/processed/dvf-stations/dvf_appartements.parquet` : une ligne par vente d'appartement nettoyée
- `data/processed/dvf-stations/dvf_appartements_rapport.csv` : nombre de ventes restantes à chaque étape de nettoyage
- `data/processed/dvf-stations/dvf_appartements_stations.parquet` : ventes + distance à la station la plus proche

Puis ouvrir les notebooks dans l'ordre (*Restart* puis *Run All*) :

1. `notebooks/01_exploration.ipynb` : description des prix selon la distance (tranches, années, communes, métro contre RER) ;
2. `notebooks/02_analyse_finale.ipynb` : corrélations, variation du prix par tranche de distance, comparaison à ventes comparables, prime de proximité, robustesse.

Les notebooks ne contiennent presque pas de code : les calculs sont dans `src/vrai_prix/analysis.py` et les graphiques dans `src/vrai_prix/visualization.py`.

## Structure du dépôt

```
.
├── data/
│   ├── raw/                 # données brutes téléchargées (ignorées par Git)
│   └── processed/dvf-stations/ # données nettoyées (ignorées par Git)
├── docs/sources.md          # sources, URL, dates, licences
├── notebooks/
│   ├── 01_exploration.ipynb     # analyse descriptive
│   └── 02_analyse_finale.ipynb  # effet de la distance, conclusions
├── outputs/
│   ├── figures/             # figures du rapport
│   └── tables/              # tableaux de résultats
├── src/vrai_prix/
│   ├── config.py            # chemins, périmètre, seuils de nettoyage
│   ├── dvf.py               # téléchargement et nettoyage du DVF
│   ├── stations.py          # stations IDFM et distance à la station la plus proche
│   ├── analysis.py          # calculs : tableaux, corrélations, comparaisons
│   └── visualization.py     # graphiques
├── tests/                   # pytest
├── pyproject.toml
└── requirements.txt
```

## Méthode

**Nettoyage du DVF** (`dvf.py`). Une ligne par vente d'appartement :

1. lignes dupliquées supprimées ;
2. ventes simples uniquement (`nature_mutation == "Vente"` : exclut VEFA, échanges, adjudications) ;
3. une seule disposition par mutation ;
4. la mutation contient exactement un appartement et aucune maison ni local commercial (les dépendances, cave ou parking, sont tolérées et comptées dans `n_dependances`) ;
5. prix, surface (9 à 300 m²) et coordonnées (emprise Île-de-France) renseignés ;
6. prix au m² entre 1 500 et 30 000 € ;
7. `nombre_pieces_principales` hors de 1 à 10 traité comme manquant.

Les seuils sont modifiables dans `src/vrai_prix/config.py`.

**Distance à la station** (`stations.py`). Les coordonnées sont projetées en Lambert-93 (EPSG:2154, en mètres). Pour chaque vente, on retient la station de métro ou de RER la plus proche **ouverte à la date de la vente** : le réseau est donc recalculé à chaque date de mise en service, ce qui permet d'étudier l'ouverture des prolongements des lignes 11 et 14 en 2024.

**Mesure de l'effet de la distance (sans modèle).** Le prix relatif d'une vente est son écart au prix médian de sa commune la même année. Pour comparer des biens comparables, on forme des « cellules » : même commune, même année, même classe de surface (moins de 30 m², 30-50, 50-80, plus de 80). Dans chaque cellule, on compare les prix médians selon la distance à la station (tranche de référence 200-400 m ; ou ventes à moins de 300 m contre ventes à plus de 600 m), puis on résume sur l'ensemble des cellules (moyenne pondérée, médiane, part de cellules positives). On y ajoute des corrélations (Spearman, Pearson) entre la distance et le prix, brut puis relatif à la commune, et la variation du prix médian par tranche de 200 m.

## Résultats

Les résultats détaillés sont dans les notebooks. Les tableaux et figures sont enregistrés dans `outputs/Tables/` et `outputs/figures/`.

- **Corrélation trompeuse à grande échelle.** Prix et distance sont fortement corrélés sur l'ensemble (Spearman ≈ −0,56 à −0,60), mais surtout parce que Paris est à la fois plus cher et mieux desservi. Rapporté au prix médian de sa commune, la corrélation tombe presque à zéro (0 à −0,06).
- **La distance la plus valorisée se situe entre 100 et 400 m.** À biens comparables, le prix est stable jusqu'à 400 m puis baisse, surtout au-delà de 600 m (environ −6,5 % à Paris et −3 % en petite couronne entre 600 et 1 000 m, par rapport à 200-400 m). En petite couronne, on observe aussi une légère décote à moins de 100 m (≈ −3,5 %), compatible avec des nuisances, mais fondée sur peu de ventes.
- **Prime de proximité en petite couronne.** Les biens à moins de 300 m d'une station se vendent en moyenne 7 à 9 % plus cher que ceux à plus de 600 m (selon l'année), à commune et surface comparables ; 5 à 9 % selon la spécification testée. L'écart typique (médiane des cellules) est plutôt de l'ordre de 5 %, car quelques cellules à gros écarts tirent la moyenne vers le haut. Ce résultat est stable de 2021 à 2025 et robuste aux seuils. À Paris, le signe est le plus souvent positif, mais trop peu de ventes éloignées existent pour le chiffrer.
- **Étude de cas : prolongements des lignes 11 et 14 (juin 2024).** Dans les quartiers qui ont gagné une station, le prix médian augmente d'environ 9 % entre 2021 et 2025, contre une baisse d'environ 7 % dans les quartiers témoins des mêmes communes : un écart de 12 à 16 points selon les seuils retenus. L'écart apparaît dès 2023, avant l'ouverture, ce qui suggère une anticipation par les acheteurs.

## Limites

- Le DVF ne contient ni étage, ni balcon, ni année de construction, ni état du bien : les ventes comparées ont la même commune et la même taille, mais pas forcément le même quartier, étage ou état.
- Les coordonnées sont celles du centre de la parcelle, pas de l'immeuble : la précision est limitée pour comparer 50 m et 100 m.
- Le prix d'éventuelles dépendances vendues avec l'appartement est inclus dans le prix au m².
- Les stations sont souvent implantées dans des zones déjà denses et recherchées : les écarts mesurés sont des associations, pas des effets causaux
- Les parcs, axes routiers, fréquentation et équipements ne sont pas pris en compte.

## Tests

```bash
python -m pytest
```

## Travail en équipe

Une branche par tâche, intégration par Pull Request relue par un autre membre. Les données DVF (brutes et nettoyées) et les stations IDFM sont versionnées dans `data/` pour que les notebooks tournent sans téléchargement (environ 100 Mo). Elles peuvent aussi être régénérées avec les commandes ci-dessous. Les chemins sont définis une seule fois, dans `src/vrai_prix/config.py` (`.gitignore`).

## Transparence sur l'usage d'un assistant

Une partie du code (téléchargement et nettoyage du DVF, tests) a été écrite avec l'aide de Claude (Anthropic), puis relue, testée et adaptée par l'équipe.

## Licences des données

DVF : Licence Ouverte (Etalab). Données Île-de-France Mobilités : voir la licence indiquée sur le portail de chaque jeu de données.

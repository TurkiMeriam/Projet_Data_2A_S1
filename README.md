# Projet_Data_2A_S1
# Le Vrai Prix de la Ville

Projet Python : combien la proximité d'une station de métro ou de RER ajoute-t-elle (ou retire-t-elle) au prix au m² des appartements à Paris et en petite couronne (75, 92, 93, 94) ?

Les ventes réelles (DVF géolocalisées) sont croisées avec la position des stations (IDFM), leur fréquentation (RATP), les parcs et axes routiers (OpenStreetMap) et les équipements de proximité (INSEE).

**Équipe :** BAHAN Karafa Ashley Kevin  · TURKI Meriam  · NAFTI Mohamed Amin 

## Installation

```bash
git clone <url-du-depot>
cd le-vrai-prix-de-la-ville
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # rend `import vrai_prix` disponible partout
```

## Utilisation rapide

```bash
# Télécharge les DVF (75, 92, 93, 94 x 5 années), nettoie, écrit le Parquet
python -m vrai_prix.dvf

# Options
python -m vrai_prix.dvf --years 2024 2025 --deps 75   # sous-ensemble
python -m vrai_prix.dvf --skip-download               # réutilise data/raw/
python -m vrai_prix.dvf --prix-min 2000 --prix-max 25000
```

Sorties (hors Git) :

- `data/raw/dvf/<année>/<dep>.csv.gz` : fichiers bruts téléchargés
- `data/processed/dvf_appartements.parquet` : une ligne par vente d'appartement
- `data/processed/dvf_appartements_rapport.csv` : nombre de mutations restantes à chaque étape de nettoyage (utile pour la section « limites » du rapport)

Dans un notebook :

```python
from vrai_prix.dvf import load_dvf
df = load_dvf()
```

Colonnes du Parquet : `id_mutation, date_mutation, annee, code_departement, code_commune, nom_commune, code_postal, adresse, id_parcelle, valeur_fonciere, surface_reelle_bati, nombre_pieces_principales, n_dependances, prix_m2, log_prix_m2, longitude, latitude`.

## Structure du dépôt

```
.
├── data/
│   ├── raw/             # fichiers téléchargés (ignorés par Git)
│   └── processed/       # Parquet nettoyés (ignorés par Git)
├── docs/                # notes de cadrage, rapport
├── notebooks/           # exploration et figures (numérotés : 01_, 02_...)
├── src/vrai_prix/
│   ├── config.py        # chemins, périmètre, seuils de nettoyage
│   └── dvf.py           # téléchargement + nettoyage DVF
├── tests/               # pytest
├── pyproject.toml
└── requirements.txt
```

Les modules suivants iront dans `src/vrai_prix/` : `stations.py` (IDFM, jointure `sjoin_nearest` en EPSG:2154), `osm.py` (parcs, axes), `model.py` (régression), `viz.py` (cartes, courbe de décote).

## Règles de nettoyage DVF

1. Lignes strictement dupliquées supprimées.
2. `nature_mutation == "Vente"` : exclut VEFA, échanges, adjudications.
3. Une seule disposition par mutation.
4. La mutation contient exactement un appartement et aucune maison ni local commercial. Les dépendances (cave, parking) sont tolérées et comptées dans `n_dependances`.
5. Prix, surface (9 à 300 m²) et coordonnées (emprise Île-de-France) renseignés.
6. Prix au m² entre 1 500 et 30 000 € (seuils de la note de cadrage, modifiables dans `config.py`).
7. `nombre_pieces_principales` hors de 1 à 10 passe à manquant (variable jugée peu fiable).

**Limites connues :**

- `valeur_fonciere` est le prix de toute la mutation : le prix de dépendances vendues avec l'appartement est inclus dans le prix au m².
- Les coordonnées sont celles du centre de la parcelle, pas de l'immeuble.
- Le DVF n'a ni étage, ni balcon, ni année de construction.

## Travailler à trois sur GitHub

- Une branche par tâche : `feat/pipeline-dvf`, `feat/regression`, `feat/viz-cartes`.
- Pull request relue par au moins un autre membre avant fusion dans `main`.
- Ne jamais committer de données (le `.gitignore` les exclut) ni de sorties de notebook volumineuses : `Kernel > Restart & Clear Output` avant commit.
- Lancer `pytest` avant chaque PR.

## Tests

```bash
pytest
```

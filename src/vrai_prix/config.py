"""Paramètres globaux du projet (chemins, périmètre, seuils de nettoyage)."""
from __future__ import annotations

from pathlib import Path

# Racine du dépôt : src/vrai_prix/config.py -> remonte de 3 niveaux
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# --- Périmètre : Paris et petite couronne ---------------------------------
DEPARTEMENTS = ["75", "92", "93", "94"]

# Cinq années glissantes disponibles dans le DVF géolocalisé (à ajuster si
# Etalab publie un nouveau millésime).
ANNEES = [2021, 2022, 2023, 2024, 2025]

# --- Source DVF géolocalisé (Etalab) --------------------------------------
# Un fichier par année et par département : bien plus léger que le fichier
# France entière (~500 Mo). Si l'URL change, ne modifier que cette constante.
DVF_BASE_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv"
DVF_URL_TEMPLATE = "{base}/{annee}/departements/{dep}.csv.gz"

# --- Seuils de nettoyage (note de cadrage, section 3) ---------------------
PRIX_M2_MIN = 1_500
PRIX_M2_MAX = 30_000
SURFACE_MIN = 9      # m² : en dessous, surface non plausible pour un logement
SURFACE_MAX = 300    # m² : au-delà, probablement une erreur ou un bien atypique
PIECES_MAX = 10      # au-delà : traité comme valeur manquante

# Emprise approximative de l'Île-de-France (WGS84) pour écarter les
# géocodages aberrants.
LAT_BOUNDS = (48.1, 49.3)
LON_BOUNDS = (1.4, 3.6)

# Fichier de sortie du pipeline DVF
DVF_CLEAN_PARQUET = DATA_PROCESSED / "dvf_appartements.parquet"

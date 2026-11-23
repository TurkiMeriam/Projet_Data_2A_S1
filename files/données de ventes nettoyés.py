"""Pipeline DVF autonome : téléchargement, nettoyage, un CSV par année.

Fichier unique, sans dépendance au package du dépôt : il se lance directement
(bouton « Run » de VS Code ou `python code_dvf.py`).

Sorties (créées à côté de ce fichier) :
    data/raw/dvf/<année>/<dep>.csv.gz       fichiers bruts téléchargés
    data/processed/dvf_appartements_<année>.csv   ventes nettoyées, une année par fichier
    data/processed/rapport_nettoyage.csv    nombre de ventes restantes à chaque étape
"""
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paramètres (à modifier ici)
# ---------------------------------------------------------------------------
ANNEES = [2021, 2022, 2023, 2024, 2025]
DEPARTEMENTS = ["75", "92", "93", "94"]

BASE_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv"
URL_TEMPLATE = "{base}/{annee}/departements/{dep}.csv.gz"

PRIX_M2_MIN, PRIX_M2_MAX = 1_500, 30_000
SURFACE_MIN, SURFACE_MAX = 9, 300
PIECES_MAX = 10
LAT_BOUNDS = (48.1, 49.3)
LON_BOUNDS = (1.4, 3.6)

DOSSIER = Path(__file__).resolve().parent
DATA_RAW = DOSSIER / "data" / "raw"
DATA_PROCESSED = DOSSIER / "data" / "processed"

log = logging.getLogger("dvf")

USECOLS = [
    "id_mutation", "date_mutation", "numero_disposition", "nature_mutation",
    "valeur_fonciere", "adresse_numero", "adresse_nom_voie", "code_postal",
    "code_commune", "nom_commune", "code_departement", "id_parcelle",
    "type_local", "surface_reelle_bati", "nombre_pieces_principales",
    "lot1_numero", "lot2_numero", "longitude", "latitude",
]
STR_COLS = {
    "id_mutation": "string", "nature_mutation": "string",
    "adresse_numero": "string", "adresse_nom_voie": "string",
    "code_postal": "string", "code_commune": "string", "nom_commune": "string",
    "code_departement": "string", "id_parcelle": "string", "type_local": "string",
    "lot1_numero": "string", "lot2_numero": "string",
}
OUTPUT_COLS = [
    "id_mutation", "date_mutation", "annee", "code_departement", "code_commune",
    "nom_commune", "code_postal", "adresse", "id_parcelle", "valeur_fonciere",
    "surface_reelle_bati", "nombre_pieces_principales", "n_dependances",
    "prix_m2", "log_prix_m2", "longitude", "latitude",
]
TYPES_A_EXCLURE = ["Maison", "Local industriel. commercial ou assimilé"]


# ---------------------------------------------------------------------------
# Téléchargement
# ---------------------------------------------------------------------------
def chemin_brut(annee, dep):
    return DATA_RAW / "dvf" / str(annee) / f"{dep}.csv.gz"


def telecharger(url, dest, tentatives=3, timeout=60):
    """Télécharge url vers dest (fichier .part puis renommage)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    derniere_erreur = None
    for essai in range(1, tentatives + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code == 404:
                    raise FileNotFoundError(f"404 : fichier introuvable {url}")
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for morceau in r.iter_content(chunk_size=1 << 20):
                        f.write(morceau)
            tmp.replace(dest)
            return dest
        except FileNotFoundError:
            raise
        except requests.RequestException as e:
            derniere_erreur = e
            log.warning("Essai %d/%d échoué pour %s : %s", essai, tentatives, url, e)
            time.sleep(2 * essai)
    raise RuntimeError(f"Échec du téléchargement de {url}") from derniere_erreur


def telecharger_tout(annees=ANNEES, deps=DEPARTEMENTS):
    for annee in annees:
        for dep in deps:
            dest = chemin_brut(annee, dep)
            if dest.exists():
                log.info("Déjà présent : %s", dest)
            else:
                url = URL_TEMPLATE.format(base=BASE_URL, annee=annee, dep=dep)
                log.info("Téléchargement %s", url)
                telecharger(url, dest)


# ---------------------------------------------------------------------------
# Lecture et nettoyage
# ---------------------------------------------------------------------------
def lire_dvf(chemin):
    return pd.read_csv(
        chemin,
        usecols=lambda c: c in USECOLS,
        dtype=STR_COLS,
        parse_dates=["date_mutation"],
    )


def nettoyer(df):
    """Renvoie (une ligne par vente d'appartement, rapport par étape)."""
    rapport = {}
    df = df.drop_duplicates().copy()
    rapport["mutations_brutes"] = df["id_mutation"].nunique()

    # 1. Ventes simples uniquement
    df = df[df["nature_mutation"] == "Vente"]
    rapport["apres_nature_vente"] = df["id_mutation"].nunique()

    # 2. Une seule disposition par mutation
    if "numero_disposition" in df.columns:
        max_disp = df.groupby("id_mutation")["numero_disposition"].transform("max")
        df = df[max_disp.fillna(1) <= 1]
    rapport["apres_disposition_unique"] = df["id_mutation"].nunique()

    # 3. Exactement un appartement, ni maison ni local commercial
    est_appart = df["type_local"] == "Appartement"
    est_autre = df["type_local"].isin(TYPES_A_EXCLURE)
    est_dep = df["type_local"] == "Dépendance"
    stats = (
        df.assign(_a=est_appart, _o=est_autre, _d=est_dep)
        .groupby("id_mutation")[["_a", "_o", "_d"]]
        .sum()
    )
    ok = stats[(stats["_a"] == 1) & (stats["_o"] == 0)]
    df = df[df["id_mutation"].isin(ok.index) & (df["type_local"] == "Appartement")]
    df = df.join(ok["_d"].rename("n_dependances"), on="id_mutation")
    rapport["apres_un_seul_appartement"] = df["id_mutation"].nunique()

    # 4. Variables indispensables renseignées et plausibles
    df = df.dropna(subset=["valeur_fonciere", "surface_reelle_bati", "longitude", "latitude"])
    df = df[
        (df["valeur_fonciere"] > 0)
        & df["surface_reelle_bati"].between(SURFACE_MIN, SURFACE_MAX)
        & df["latitude"].between(*LAT_BOUNDS)
        & df["longitude"].between(*LON_BOUNDS)
    ]
    rapport["apres_variables_plausibles"] = df["id_mutation"].nunique()

    # 5. Prix au m² et valeurs aberrantes
    df = df.assign(prix_m2=df["valeur_fonciere"] / df["surface_reelle_bati"])
    df = df[df["prix_m2"].between(PRIX_M2_MIN, PRIX_M2_MAX)]
    rapport["apres_filtre_prix_m2"] = df["id_mutation"].nunique()

    # 6. Variables dérivées
    pieces = df["nombre_pieces_principales"].astype("float")
    pieces = pieces.where(pieces.between(1, PIECES_MAX))
    df = df.assign(
        nombre_pieces_principales=pieces.astype("Int64"),
        n_dependances=df["n_dependances"].astype(int),
        annee=df["date_mutation"].dt.year.astype("Int64"),
        log_prix_m2=np.log(df["prix_m2"]),
        adresse=(
            df["adresse_numero"].fillna("").astype(str) + " "
            + df["adresse_nom_voie"].fillna("").astype(str)
        ).str.strip(),
    )

    df = df.drop_duplicates(subset="id_mutation")
    rapport["mutations_finales"] = len(df)

    sortie = df[OUTPUT_COLS].sort_values(["date_mutation", "id_mutation"]).reset_index(drop=True)
    return sortie, pd.Series(rapport, name="n_mutations")


# ---------------------------------------------------------------------------
# Export : un CSV par année
# ---------------------------------------------------------------------------
def exporter_csv_par_annee(df, dossier=DATA_PROCESSED):
    dossier.mkdir(parents=True, exist_ok=True)
    chemins = []
    for annee, sous_df in df.groupby("annee"):
        chemin = dossier / f"dvf_appartements_{int(annee)}.csv"
        sous_df.to_csv(chemin, index=False)
        log.info("Écrit : %s (%d lignes)", chemin, len(sous_df))
        chemins.append(chemin)
    return chemins


def main(telechargement=True):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if telechargement:
        telecharger_tout()

    morceaux, rapports = [], []
    for annee in ANNEES:
        for dep in DEPARTEMENTS:
            chemin = chemin_brut(annee, dep)
            if not chemin.exists():
                raise FileNotFoundError(f"{chemin} absent : lancer avec telechargement=True")
            propre, rapport = nettoyer(lire_dvf(chemin))
            morceaux.append(propre)
            rapports.append(rapport)
            log.info("%s / %s : %d ventes retenues", annee, dep, len(propre))

    resultat = pd.concat(morceaux, ignore_index=True)
    exporter_csv_par_annee(resultat)

    total = pd.concat(rapports, axis=1).sum(axis=1).astype(int)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    total.to_csv(DATA_PROCESSED / "rapport_nettoyage.csv", header=["n_mutations"])
    log.info("Rapport de nettoyage :\n%s", total.to_string())

    # Aperçu : 4 premières lignes
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    print(resultat.shape)
    print(resultat.head(4))
    return resultat


if __name__ == "__main__":
    main()

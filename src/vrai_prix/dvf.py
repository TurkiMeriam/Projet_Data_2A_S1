"""Téléchargement et nettoyage des DVF géolocalisées (Etalab).

Pipeline : download_dvf -> read_dvf -> clean_dvf -> build_dataset (Parquet).

Usage en ligne de commande (depuis la racine du dépôt) :

    python -m vrai_prix.dvf                       # tout le périmètre du projet
    python -m vrai_prix.dvf --years 2024 2025 --deps 75
    python -m vrai_prix.dvf --skip-download       # réutilise data/raw/
Choix de nettoyage (une ligne = une vente d'un appartement) :
  1. suppression des lignes strictement dupliquées (le DVF répète une ligne
     par nature de culture ; les numéros de lot évitent de fusionner deux
     logements distincts) ;
  2. nature_mutation == "Vente" (exclut notamment les VEFA, échanges, etc.) ;
  3. une seule disposition par mutation ;
  4. la mutation contient exactement un appartement et aucune maison ni local
     commercial/industriel (les dépendances : cave, parking... sont tolérées) ;
  5. prix, surface, coordonnées renseignés et plausibles ;
  6. prix au m² compris entre PRIX_M2_MIN et PRIX_M2_MAX.

Limite connue : valeur_fonciere est le prix de toute la mutation. Si des
dépendances sont vendues avec l'appartement, leur prix est inclus dans le prix
au m² (colonne n_dependances pour les repérer / les écarter si besoin).
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

# Colonnes du DVF géolocalisé réellement utilisées (le reste est ignoré à la
# lecture pour économiser la mémoire).
USECOLS = [
    "id_mutation", "date_mutation", "numero_disposition", "nature_mutation",
    "valeur_fonciere", "adresse_numero", "adresse_nom_voie", "code_postal",
    "code_commune", "nom_commune", "code_departement", "id_parcelle",
    "type_local", "surface_reelle_bati", "nombre_pieces_principales",
    "lot1_numero", "lot2_numero",  # distinguent deux logements identiques d'une même vente
    "longitude", "latitude",
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
def dvf_url(annee: int, dep: str, base: str = config.DVF_BASE_URL) -> str:
    return config.DVF_URL_TEMPLATE.format(base=base, annee=annee, dep=dep)


def raw_path(annee: int, dep: str, raw_dir: Path = config.DATA_RAW) -> Path:
    return Path(raw_dir) / "dvf" / str(annee) / f"{dep}.csv.gz"


def download_file(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> Path:
    """Télécharge `url` vers `dest` (écriture atomique via un fichier .part)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code == 404:
                    raise FileNotFoundError(
                        f"404 sur {url} : millésime ou URL introuvable. "
                        "Vérifier config.ANNEES / config.DVF_BASE_URL."
                    )
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            return dest
        except FileNotFoundError:
            raise
        except requests.RequestException as e:  # réseau / HTTP transitoire
            last_err = e
            log.warning("Tentative %d/%d échouée pour %s : %s", attempt, retries, url, e)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Échec du téléchargement de {url}") from last_err


def download_dvf(
    years: Iterable[int] = config.ANNEES,
    deps: Iterable[str] = config.DEPARTEMENTS,
    raw_dir: Path = config.DATA_RAW,
    force: bool = False,
) -> list[Path]:
    """Télécharge un fichier par (année, département) ; saute ceux déjà présents."""
    paths = []
    for annee in years:
        for dep in deps:
            dest = raw_path(annee, dep, raw_dir)
            if dest.exists() and not force:
                log.info("Déjà présent : %s", dest)
            else:
                log.info("Téléchargement %s", dvf_url(annee, dep))
                download_file(dvf_url(annee, dep), dest)
            paths.append(dest)
    return paths


# ---------------------------------------------------------------------------
# Lecture et nettoyage
# ---------------------------------------------------------------------------
def read_dvf(path: str | Path) -> pd.DataFrame:
    """Lit un CSV(.gz) DVF géolocalisé en ne gardant que les colonnes utiles."""
    return pd.read_csv(
        path,
        usecols=lambda c: c in USECOLS,
        dtype=STR_COLS,
        parse_dates=["date_mutation"],
    )


def clean_dvf(
    df: pd.DataFrame,
    prix_min: float = config.PRIX_M2_MIN,
    prix_max: float = config.PRIX_M2_MAX,
    surface_min: float = config.SURFACE_MIN,
    surface_max: float = config.SURFACE_MAX,
    pieces_max: int = config.PIECES_MAX,
) -> tuple[pd.DataFrame, pd.Series]:
    """Nettoie un DVF brut : renvoie (une ligne par vente d'appartement, rapport).

    Le rapport donne le nombre de mutations restantes à chaque étape, utile
    pour la section « limites des données » du rapport final.
    """
    report: dict[str, int] = {}
    df = df.drop_duplicates().copy()
    report["mutations_brutes"] = df["id_mutation"].nunique()

    # 1. Ventes simples uniquement (exclut VEFA, échange, adjudication...)
    df = df[df["nature_mutation"] == "Vente"]
    report["apres_nature_vente"] = df["id_mutation"].nunique()

    # 2. Une seule disposition par mutation
    if "numero_disposition" in df.columns:
        max_disp = df.groupby("id_mutation")["numero_disposition"].transform("max")
        df = df[max_disp.fillna(1) <= 1]
    report["apres_disposition_unique"] = df["id_mutation"].nunique()

    # 3. Exactement un appartement, aucune maison / local commercial
    is_appart = df["type_local"] == "Appartement"
    is_autre = df["type_local"].isin(TYPES_A_EXCLURE)
    is_dep = df["type_local"] == "Dépendance"
    g = df.assign(_a=is_appart, _o=is_autre, _d=is_dep).groupby("id_mutation")
    stats = g[["_a", "_o", "_d"]].sum()
    ok = stats[(stats["_a"] == 1) & (stats["_o"] == 0)]
    df = df[df["id_mutation"].isin(ok.index) & (df["type_local"] == "Appartement")]
    df = df.join(ok["_d"].rename("n_dependances"), on="id_mutation")
    report["apres_un_seul_appartement"] = df["id_mutation"].nunique()

    # 4. Variables indispensables renseignées et plausibles
    df = df.dropna(subset=["valeur_fonciere", "surface_reelle_bati", "longitude", "latitude"])
    df = df[
        (df["valeur_fonciere"] > 0)
        & df["surface_reelle_bati"].between(surface_min, surface_max)
        & df["latitude"].between(*config.LAT_BOUNDS)
        & df["longitude"].between(*config.LON_BOUNDS)
    ]
    report["apres_variables_plausibles"] = df["id_mutation"].nunique()

    # 5. Prix au m² et valeurs aberrantes
    df = df.assign(prix_m2=df["valeur_fonciere"] / df["surface_reelle_bati"])
    df = df[df["prix_m2"].between(prix_min, prix_max)]
    report["apres_filtre_prix_m2"] = df["id_mutation"].nunique()

    # 6. Variables dérivées
    pieces = df["nombre_pieces_principales"].astype("float")
    pieces = pieces.where(pieces.between(1, pieces_max))  # 0 ou >max -> manquant
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

    # Une ligne par mutation (garde-fou : un appartement vendu sur 2 parcelles)
    df = df.drop_duplicates(subset="id_mutation")
    report["mutations_finales"] = len(df)

    out = df[OUTPUT_COLS].sort_values(["date_mutation", "id_mutation"]).reset_index(drop=True)
    return out, pd.Series(report, name="n_mutations")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_dataset(
    years: Iterable[int] = config.ANNEES,
    deps: Iterable[str] = config.DEPARTEMENTS,
    raw_dir: Path = config.DATA_RAW,
    out_path: Path = config.DVF_CLEAN_PARQUET,
    download: bool = True,
    force: bool = False,
    **clean_kwargs,
) -> pd.DataFrame:
    """Télécharge (optionnel), nettoie chaque fichier, écrit le Parquet final."""
    years, deps = list(years), list(deps)
    if download:
        download_dvf(years, deps, raw_dir, force=force)

    parts, reports = [], []
    for annee in years:
        for dep in deps:
            path = raw_path(annee, dep, raw_dir)
            if not path.exists():
                raise FileNotFoundError(f"{path} absent : lancer sans --skip-download.")
            clean, rep = clean_dvf(read_dvf(path), **clean_kwargs)
            parts.append(clean)
            reports.append(rep)
            log.info("%s / %s : %d ventes retenues", annee, dep, len(clean))

    result = pd.concat(parts, ignore_index=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)

    total = pd.concat(reports, axis=1).sum(axis=1).astype(int)
    total.to_csv(out_path.with_name(out_path.stem + "_rapport.csv"), header=["n_mutations"])
    log.info("Rapport de nettoyage :\n%s", total.to_string())
    log.info("Écrit : %s (%d lignes)", out_path, len(result))
    return result


def load_dvf(path: Path = config.DVF_CLEAN_PARQUET) -> pd.DataFrame:
    """Charge le Parquet nettoyé (à utiliser dans les notebooks)."""
    return pd.read_parquet(path)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", default=config.ANNEES)
    p.add_argument("--deps", nargs="+", default=config.DEPARTEMENTS)
    p.add_argument("--skip-download", action="store_true", help="utiliser data/raw/ tel quel")
    p.add_argument("--force", action="store_true", help="retélécharger même si présent")
    p.add_argument("--prix-min", type=float, default=config.PRIX_M2_MIN)
    p.add_argument("--prix-max", type=float, default=config.PRIX_M2_MAX)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_dataset(
        years=args.years, deps=args.deps, download=not args.skip_download,
        force=args.force, prix_min=args.prix_min, prix_max=args.prix_max,
    )


if __name__ == "__main__":
    main()

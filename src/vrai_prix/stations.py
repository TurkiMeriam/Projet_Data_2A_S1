"""Stations de métro / RER (IDFM) et distance de chaque vente DVF à la plus proche.

Pipeline : download_stations -> read_stations_lignes -> add_station_features.

Usage en ligne de commande (depuis la racine du dépôt, après `python -m vrai_prix.dvf`) :

    python -m vrai_prix.stations                  # télécharge + joint
    python -m vrai_prix.stations --skip-download  # réutilise data/raw/stations/
    python -m vrai_prix.stations --modes METRO RER TRAMWAY

Choix de méthode :
  1. Le fichier IDFM a une ligne par couple (gare, ligne). On regroupe par zone de
     correspondance (id_ref_zdc) : Châtelet = une station desservie par plusieurs
     lignes. Sans ce regroupement, une même station apparaîtrait plusieurs fois
     (ex aequo dans la recherche du plus proche voisin).
  2. Une ligne n'est comptée à une gare qu'à partir de sa date de mise en service
     (date_mise). Une vente de 2023 près de Villejuif - Gustave Roussy n'est pas
     rattachée à la ligne 14, ouverte en juin 2024. Concrètement, on découpe la
     période en tranches entre deux ouvertures et on fait une jointure par tranche.
  3. Les distances sont calculées en Lambert 93 (EPSG:2154), donc en mètres.

Limite : le fichier décrit le réseau actuel. Une gare fermée avant aujourd'hui en
serait absente (aucun cas connu sur Paris + petite couronne en 2021-2025).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd

from . import config
from .dvf import download_file, load_dvf

log = logging.getLogger(__name__)

# Colonnes IDFM utilisées -> noms dans le projet
COLS_IDFM = {
    "id_ref_zdc": "id_station",
    "nom_zdc": "nom_station",
    "res_com": "ligne",        # ex. "METRO 14", "RER A"
    "mode": "mode",
    "date_mise": "date_mise",
}

STATION_COLS = [
    "id_station", "nom_station", "lignes_station", "modes_station",
    "nb_lignes_station", "date_ouverture_station", "dist_station_m",
    "classe_dist", "nb_stations_voisinage",
]


# ---------------------------------------------------------------------------
# Téléchargement et lecture
# ---------------------------------------------------------------------------
def download_stations(dest: Path = config.STATIONS_RAW, force: bool = False) -> Path:
    """Télécharge l'export GeoJSON IDFM (saute si déjà présent)."""
    dest = Path(dest)
    if dest.exists() and not force:
        log.info("Déjà présent : %s", dest)
        return dest
    log.info("Téléchargement %s", config.STATIONS_URL)
    return download_file(config.STATIONS_URL, dest)


def read_stations_lignes(
    path: str | Path = config.STATIONS_RAW,
    modes: Iterable[str] = config.MODES_RETENUS,
) -> gpd.GeoDataFrame:
    """Une ligne par couple (station, ligne), en Lambert 93.

    Colonnes : id_station, nom_station, ligne, mode, date_mise, geometry.
    """
    gdf = gpd.read_file(path)
    return prepare_stations_lignes(gdf, modes)


def prepare_stations_lignes(
    gdf: gpd.GeoDataFrame, modes: Iterable[str] = config.MODES_RETENUS
) -> gpd.GeoDataFrame:
    """Filtre les modes, renomme, type les dates et projette en mètres."""
    gdf = gdf[gdf["mode"].isin(list(modes)) & gdf.geometry.notna()]
    gdf = gdf[list(COLS_IDFM) + ["geometry"]].rename(columns=COLS_IDFM)
    # date_mise arrive en texte ISO avec fuseau ("2024-06-24T00:00:00+00:00") :
    # on la ramène à une date sans fuseau, comparable à date_mutation du DVF.
    gdf["date_mise"] = (
        pd.to_datetime(gdf["date_mise"], utc=True, errors="coerce")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    gdf["id_station"] = gdf["id_station"].astype("Int64").astype("string")
    gdf = gdf.drop_duplicates(subset=["id_station", "ligne"])
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(config.CRS_METRIQUE).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Réseau ouvert à une date donnée
# ---------------------------------------------------------------------------
def stations_ouvertes(lignes: gpd.GeoDataFrame, date: pd.Timestamp) -> gpd.GeoDataFrame:
    """Stations (une ligne par zone de correspondance) ouvertes à `date`.

    Une date de mise en service manquante est traitée comme ancienne (ouverte).
    La position d'une station est le centre des quais de ses lignes ouvertes.
    """
    ouvertes = lignes[lignes["date_mise"].isna() | (lignes["date_mise"] <= date)]
    xy = ouvertes.assign(x=ouvertes.geometry.x, y=ouvertes.geometry.y)
    agg = xy.groupby("id_station").agg(
        nom_station=("nom_station", "first"),
        lignes_station=("ligne", lambda s: ", ".join(sorted(s.unique()))),
        modes_station=("mode", lambda s: "+".join(sorted(s.unique()))),
        nb_lignes_station=("ligne", "nunique"),
        date_ouverture_station=("date_mise", "min"),
        x=("x", "mean"),
        y=("y", "mean"),
    )
    geom = gpd.points_from_xy(agg["x"], agg["y"], crs=config.CRS_METRIQUE)
    return gpd.GeoDataFrame(
        agg.drop(columns=["x", "y"]).reset_index(), geometry=geom, crs=config.CRS_METRIQUE
    )


def dates_ouverture(lignes: gpd.GeoDataFrame, debut, fin) -> list[pd.Timestamp]:
    """Dates de mise en service strictement comprises dans ]debut, fin]."""
    d = lignes["date_mise"].dropna().unique()
    return sorted(pd.Timestamp(x) for x in d if debut < x <= fin)


# ---------------------------------------------------------------------------
# Jointure avec le DVF
# ---------------------------------------------------------------------------
def dvf_to_gdf(dvf: pd.DataFrame) -> gpd.GeoDataFrame:
    """Points DVF (WGS84 -> Lambert 93)."""
    pts = gpd.points_from_xy(dvf["longitude"], dvf["latitude"], crs=4326)
    return gpd.GeoDataFrame(dvf, geometry=pts).to_crs(config.CRS_METRIQUE)


def _join_snapshot(
    ventes: gpd.GeoDataFrame, stations: gpd.GeoDataFrame, rayon: float
) -> pd.DataFrame:
    """Plus proche station + nombre de stations dans le rayon, pour un état du réseau."""
    near = gpd.sjoin_nearest(
        ventes[["geometry"]], stations, how="left", distance_col="dist_station_m"
    )
    # Deux stations exactement à la même distance : on garde la première
    near = near[~near.index.duplicated(keep="first")]

    # Comptage des stations à moins de `rayon` mètres (index spatial)
    i_vente, _ = stations.sindex.query(ventes.geometry, predicate="dwithin", distance=rayon)
    nb = np.bincount(i_vente, minlength=len(ventes))

    out = pd.DataFrame(near.drop(columns=["geometry", "index_right"]))
    out["nb_stations_voisinage"] = nb
    return out


def add_station_features(
    dvf: pd.DataFrame,
    lignes: gpd.GeoDataFrame,
    classes: list[float] = config.CLASSES_DIST,
    rayon: float = config.RAYON_VOISINAGE,
) -> pd.DataFrame:
    """Ajoute au DVF les caractéristiques de la station la plus proche à la date de vente.

    Colonnes ajoutées : voir STATION_COLS. Le nombre de lignes du DVF est inchangé.
    """
    dvf = dvf.reset_index(drop=True)
    ventes = dvf_to_gdf(dvf)
    dates = pd.to_datetime(dvf["date_mutation"])

    # Tranches de temps entre deux ouvertures : le réseau est constant dans chacune
    debut, fin = dates.min(), dates.max()
    coupures = dates_ouverture(lignes, debut, fin)
    bornes = [debut] + coupures
    tranche = np.searchsorted(np.array(coupures, dtype="datetime64[ns]"),
                              dates.to_numpy(), side="right")

    morceaux = []
    for k, date_reseau in enumerate(bornes):
        masque = tranche == k
        if not masque.any():
            continue
        stations = stations_ouvertes(lignes, date_reseau)
        morceaux.append(_join_snapshot(ventes[masque], stations, rayon))
        log.info("Réseau au %s : %d stations, %d ventes", date_reseau.date(),
                 len(stations), masque.sum())

    feats = pd.concat(morceaux).sort_index()
    feats["classe_dist"] = pd.cut(
        feats["dist_station_m"], bins=classes, right=False,
        labels=_labels(classes),
    )
    out = dvf.join(feats[STATION_COLS])
    assert len(out) == len(dvf), "la jointure a modifié le nombre de ventes"
    return out


def _labels(classes: list[float]) -> list[str]:
    labs = []
    for a, b in zip(classes[:-1], classes[1:]):
        labs.append(f"{int(a)}+ m" if np.isinf(b) else f"{int(a)}-{int(b)} m")
    return labs


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_dvf_stations(
    dvf_path: Path = config.DVF_CLEAN_PARQUET,
    stations_path: Path = config.STATIONS_RAW,
    out_path: Path = config.DVF_STATIONS_PARQUET,
    modes: Iterable[str] = config.MODES_RETENUS,
    download: bool = True,
) -> pd.DataFrame:
    """DVF nettoyé + variables de station -> Parquet."""
    if download:
        download_stations(stations_path)
    lignes = read_stations_lignes(stations_path, modes)
    res = add_station_features(load_dvf(dvf_path), lignes)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out_path, index=False)
    log.info("Écrit : %s (%d lignes)", out_path, len(res))
    return res


def load_dvf_stations(path: Path = config.DVF_STATIONS_PARQUET) -> pd.DataFrame:
    """Charge le DVF enrichi (à utiliser dans les notebooks)."""
    return pd.read_parquet(path)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", nargs="+", default=list(config.MODES_RETENUS))
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_dvf_stations(modes=args.modes, download=not args.skip_download)


if __name__ == "__main__":
    main()

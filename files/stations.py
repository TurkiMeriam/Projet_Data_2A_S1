"""Stations de métro / RER / tramway (IDFM) et distance de chaque vente DVF.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import re
import time
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

log = logging.getLogger("stations")

# ---------------------------------------------------------------------------
# Dossiers et récupération éventuelle de config.py s'il existe dans le dépôt
# ---------------------------------------------------------------------------
DOSSIER = Path(__file__).resolve().parent
RACINE_PROJET = DOSSIER.parent if DOSSIER.name == "files" else DOSSIER

_cfg = None
for _candidat in [
    DOSSIER / "config.py",
    RACINE_PROJET / "config.py",
    *RACINE_PROJET.glob("*/config.py"),
]:
    if _candidat.exists():
        _spec = importlib.util.spec_from_file_location("_projet_config", _candidat)
        if _spec and _spec.loader:
            _cfg = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_cfg)
            break

# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------
DATA_RAW = Path(getattr(_cfg, "DATA_RAW", RACINE_PROJET / "data" / "raw"))
DATA_PROCESSED = Path(getattr(_cfg, "DATA_PROCESSED", RACINE_PROJET / "data" / "processed"))

STATIONS_URL = getattr(
    _cfg,
    "STATIONS_URL",
    "https://data.iledefrance-mobilites.fr/api/explore/v2.1/catalog/datasets/"
    "emplacement-des-gares-idf/exports/geojson",
)
STATIONS_RAW = DATA_RAW / "stations" / "emplacement_gares_idfm.geojson"
DVF_STATIONS_OUT = DATA_PROCESSED / "dvf_appartements_stations.parquet"

MODES_RETENUS = list(getattr(_cfg, "MODES_RETENUS", ["METRO", "RER"]))
CRS_METRIQUE = getattr(_cfg, "CRS_METRIQUE", "EPSG:2154")
CLASSES_DIST = list(getattr(_cfg, "CLASSES_DIST", [0, 100, 200, 300, 400, 500, np.inf]))
RAYON_VOISINAGE = float(getattr(_cfg, "RAYON_VOISINAGE", 500.0))

COLS_IDFM = {
    "id_ref_zdc": "id_station",
    "nom_zdc": "nom_station",
    "res_com": "ligne",        # ex. "METRO 14", "RER A"
    "mode": "mode",
    "date_mise": "date_mise",
}

STATION_COLS = [
    "id_station",
    "nom_station",
    "lignes_station",
    "modes_station",
    "nb_lignes_station",
    "date_ouverture_station",
    "dist_station_m",
    "classe_dist",
    "nb_stations_voisinage",
]


# ---------------------------------------------------------------------------
# Téléchargement et chargement DVF
# ---------------------------------------------------------------------------
def telecharger(url: str, dest: Path, tentatives: int = 3, timeout: int = 60) -> Path:
    """Télécharge url vers dest (fichier .part puis renommage atomique)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    derniere_erreur: Exception | None = None

    for essai in range(1, tentatives + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for morceau in r.iter_content(chunk_size=1 << 20):
                        if morceau:
                            f.write(morceau)
            tmp.replace(dest)
            return dest
        except requests.RequestException as e:
            derniere_erreur = e
            log.warning("Essai %d/%d échoué pour %s : %s", essai, tentatives, url, e)
            time.sleep(2 * essai)

    raise RuntimeError(f"Échec du téléchargement de {url}") from derniere_erreur


def download_stations(dest: Path = STATIONS_RAW, force: bool = False) -> Path:
    """Télécharge l'export GeoJSON IDFM (saute si déjà présent)."""
    dest = Path(dest)
    if dest.exists() and not force:
        log.info("Déjà présent : %s", dest)
        return dest
    log.info("Téléchargement %s", STATIONS_URL)
    return telecharger(STATIONS_URL, dest)


def charger_dvf_nettoye(dossier_processed: Path = DATA_PROCESSED) -> pd.DataFrame:
    """Charge le DVF nettoyé (Parquet s'il existe, sinon les CSV annuels dvf_appartements_AAAA.csv)."""
    dossiers = []
    for d in [Path(dossier_processed), DOSSIER / "data" / "processed", RACINE_PROJET / "data" / "processed"]:
        d_res = d.resolve()
        if d_res not in dossiers:
            dossiers.append(d_res)

    for dossier in dossiers:
        if not dossier.exists():
            continue

        parquet = dossier / "dvf_appartements_clean.parquet"
        if parquet.exists():
            df = pd.read_parquet(parquet)
            log.info("DVF chargé depuis %s : %d lignes", parquet, len(df))
            return df

        fichiers = sorted(
            f for f in dossier.glob("dvf_appartements_*.csv")
            if re.fullmatch(r"dvf_appartements_\d{4}\.csv", f.name)
        )
        if fichiers:
            morceaux = [pd.read_csv(f, parse_dates=["date_mutation"]) for f in fichiers]
            morceaux = [m for m in morceaux if not m.empty]
            if morceaux:
                res = pd.concat(morceaux, ignore_index=True)
                log.info("DVF chargé depuis %s : %d lignes (%d fichiers)", dossier, len(res), len(morceaux))
                return res

    raise FileNotFoundError(
        "Aucun fichier DVF trouvé dans : " + ", ".join(str(d) for d in dossiers)
    )


# ---------------------------------------------------------------------------
# Lecture et préparation des stations
# ---------------------------------------------------------------------------
def read_stations_lignes(
    path: str | Path = STATIONS_RAW,
    modes: Iterable[str] = MODES_RETENUS,
    crs_metrique: str = CRS_METRIQUE,
) -> gpd.GeoDataFrame:
    """Une ligne par couple (station, ligne), en Lambert 93."""
    gdf = gpd.read_file(path)
    return prepare_stations_lignes(gdf, modes=modes, crs_metrique=crs_metrique)


def prepare_stations_lignes(
    gdf: gpd.GeoDataFrame,
    modes: Iterable[str] = MODES_RETENUS,
    crs_metrique: str = CRS_METRIQUE,
) -> gpd.GeoDataFrame:
    """Filtre les modes, renomme, type les dates et projette en mètres."""
    gdf = gdf[gdf["mode"].isin(list(modes)) & gdf.geometry.notna()].copy()
    gdf = gdf[list(COLS_IDFM) + ["geometry"]].rename(columns=COLS_IDFM)

    gdf["date_mise"] = (
        pd.to_datetime(gdf["date_mise"], utc=True, errors="coerce")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    gdf["id_station"] = gdf["id_station"].astype("Int64").astype("string")
    gdf = gdf.dropna(subset=["id_station"]).drop_duplicates(subset=["id_station", "ligne"])

    if gdf.empty:
        raise ValueError("Aucune station retenue : vérifie le fichier téléchargé et les modes demandés.")

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(crs_metrique).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Réseau ouvert à une date donnée
# ---------------------------------------------------------------------------
def stations_ouvertes(
    lignes: gpd.GeoDataFrame,
    date: pd.Timestamp,
    crs_metrique: str = CRS_METRIQUE,
) -> gpd.GeoDataFrame:
    """Stations (une ligne par zone de correspondance) ouvertes à `date`."""
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
    geom = gpd.points_from_xy(agg["x"], agg["y"], crs=crs_metrique)
    return gpd.GeoDataFrame(
        agg.drop(columns=["x", "y"]).reset_index(),
        geometry=geom,
        crs=crs_metrique,
    )


def dates_ouverture(lignes: gpd.GeoDataFrame, debut: pd.Timestamp, fin: pd.Timestamp) -> list[pd.Timestamp]:
    """Dates de mise en service strictement comprises dans ]debut, fin]."""
    d = lignes["date_mise"].dropna().unique()
    return sorted(pd.Timestamp(x) for x in d if debut < x <= fin)


# ---------------------------------------------------------------------------
# Jointure avec le DVF
# ---------------------------------------------------------------------------
def dvf_to_gdf(dvf: pd.DataFrame, crs_metrique: str = CRS_METRIQUE) -> gpd.GeoDataFrame:
    """Points DVF (WGS84 -> Lambert 93)."""
    pts = gpd.points_from_xy(dvf["longitude"], dvf["latitude"], crs=4326)
    return gpd.GeoDataFrame(dvf, geometry=pts, crs=4326).to_crs(crs_metrique)


def _join_snapshot(
    ventes: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    rayon: float,
) -> pd.DataFrame:
    """Plus proche station + nombre de stations dans le rayon, pour un état du réseau."""
    near = gpd.sjoin_nearest(
        ventes[["geometry"]], stations, how="left", distance_col="dist_station_m"
    )
    near = near[~near.index.duplicated(keep="first")]

    i_vente, _ = stations.sindex.query(ventes.geometry, predicate="dwithin", distance=rayon)
    nb = np.bincount(i_vente, minlength=len(ventes))

    out = pd.DataFrame(near.drop(columns=["geometry", "index_right"]))
    out["nb_stations_voisinage"] = nb
    return out


def _labels(classes: list[float]) -> list[str]:
    labs = []
    for a, b in zip(classes[:-1], classes[1:]):
        labs.append(f"{int(a)}+ m" if np.isinf(b) else f"{int(a)}-{int(b)} m")
    return labs


def add_station_features(
    dvf: pd.DataFrame,
    lignes: gpd.GeoDataFrame,
    classes: list[float] = CLASSES_DIST,
    rayon: float = RAYON_VOISINAGE,
    crs_metrique: str = CRS_METRIQUE,
) -> pd.DataFrame:
    """Ajoute au DVF les caractéristiques de la station la plus proche à la date de vente."""
    dvf = dvf.reset_index(drop=True)
    ventes = dvf_to_gdf(dvf, crs_metrique=crs_metrique)
    dates = pd.to_datetime(dvf["date_mutation"])

    debut, fin = dates.min(), dates.max()
    coupures = dates_ouverture(lignes, debut, fin)
    bornes = [debut] + coupures
    tranche = np.searchsorted(
        np.array(coupures, dtype="datetime64[ns]"),
        dates.to_numpy(dtype="datetime64[ns]"),
        side="right",
    )

    morceaux = []
    for k, date_reseau in enumerate(bornes):
        masque = tranche == k
        if not masque.any():
            continue
        stations = stations_ouvertes(lignes, date_reseau, crs_metrique=crs_metrique)
        morceaux.append(_join_snapshot(ventes[masque], stations, rayon))
        log.info(
            "Réseau au %s : %d stations, %d ventes",
            date_reseau.date(),
            len(stations),
            masque.sum(),
        )

    feats = pd.concat(morceaux).sort_index()
    feats["classe_dist"] = pd.cut(
        feats["dist_station_m"],
        bins=classes,
        right=False,
        labels=_labels(classes),
    )
    out = dvf.join(feats[STATION_COLS])
    assert len(out) == len(dvf), "la jointure a modifié le nombre de ventes"
    return out


# ---------------------------------------------------------------------------
# Exécution principale
# ---------------------------------------------------------------------------
def main(telechargement: bool = True, modes: Iterable[str] = MODES_RETENUS) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--modes", nargs="+", default=None)
    parser.add_argument("--skip-download", action="store_true")
    args, _ = parser.parse_known_args()

    modes_actifs = args.modes if args.modes is not None else list(modes)
    faire_telechargement = False if args.skip_download else telechargement

    if faire_telechargement:
        download_stations(STATIONS_RAW)

    lignes = read_stations_lignes(STATIONS_RAW, modes=modes_actifs)
    dvf = charger_dvf_nettoye(DATA_PROCESSED)
    resultat = add_station_features(dvf, lignes)

    DVF_STATIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    resultat.to_parquet(DVF_STATIONS_OUT, index=False)
    log.info("Écrit : %s (%d lignes)", DVF_STATIONS_OUT, len(resultat))

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    print(resultat.shape)
    print(
        resultat[["date_mutation", "nom_station", "lignes_station", "dist_station_m", "classe_dist"]]
        .head(5)
    )
    return resultat


if __name__ == "__main__":
    main()

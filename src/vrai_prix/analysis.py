"""Analyse descriptive du prix au m² selon la distance à la station (sans modèle).

Les fonctions prennent un DataFrame de ventes enrichi par ``vrai_prix.stations``
(colonnes ``dist_station_m``, ``prix_m2``, ``zone``, ``annee``...) et renvoient
des DataFrames prêts à être affichés ou sauvegardés.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

TABLES = config.ROOT / "outputs" / "Tables"

ZONES = ["Paris", "Petite couronne"]

# Tranches de la note de cadrage (statistiques descriptives)
TRANCHES_NOTE = [0, 100, 300, 500, np.inf]
LABELS_NOTE = ["0-100 m", "100-300 m", "300-500 m", ">500 m"]

# --------------------------------------------------------------------------- données

def charger_donnees() -> pd.DataFrame:
    nom = "dvf_appartements_stations"
    dossiers = [config.DATA_PROCESSED, config.ROOT / "data" / "processed"]
    chemins = [d / nom for d in dossiers]

    fichier = next((p.with_suffix(".parquet") for p in chemins if p.with_suffix(".parquet").exists()), None)
    if fichier is not None:
        df = pd.read_parquet(fichier)
    else:
        fichier = next((p.with_suffix(".csv") for p in chemins if p.with_suffix(".csv").exists()), None)
        if fichier is None:
            attendus = "\n".join(str(p.with_suffix(".parquet")) for p in chemins)
            raise FileNotFoundError(
                "Fichier dvf_appartements_stations.parquet introuvable. Chemins vérifiés :\n"
                f"{attendus}\nLancer depuis la racine du projet :\n"
                "  python -m vrai_prix.dvf\n  python -m vrai_prix.stations"
            )
        df = pd.read_csv(fichier, low_memory=False)
    df["date_mutation"] = pd.to_datetime(df["date_mutation"])
    df["annee"] = df["date_mutation"].dt.year
    # codes lus parfois comme des nombres (75 ou 75.0) : on les remet en texte
    df["code_departement"] = df["code_departement"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
    df["code_commune"] = df["code_commune"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["zone"] = np.where(df["code_departement"] == "75", "Paris", "Petite couronne")
    df["commune_id"] = df["code_departement"] + "_" + df["code_commune"]
    return df


def rapport_nettoyage() -> pd.DataFrame | None:
    """Tableau « mutations restantes à chaque étape », s'il a été produit par dvf.py."""
    chemin = config.DVF_CLEAN_PARQUET.with_name("dvf_appartements_rapport.csv")
    if not chemin.exists():
        return None
    r = pd.read_csv(chemin, index_col=0)["n_mutations"].to_frame()
    r["perte"] = -r["n_mutations"].diff().fillna(0).astype(int)
    r["part_conservee_%"] = (100 * r["n_mutations"] / r["n_mutations"].iloc[0]).round(1)
    return r


def sauver(tableau: pd.DataFrame, nom: str, index: bool = True) -> pd.DataFrame:
    """Enregistre un tableau dans outputs/Tables et le renvoie (pour l'afficher dans la foulée)."""
    TABLES.mkdir(parents=True, exist_ok=True)
    tableau.to_csv(TABLES / nom, index=index)
    return tableau


# --------------------------------------------------------------------------- descriptif

def prix_par_zone_annee(df: pd.DataFrame) -> pd.DataFrame:
    """Prix médian au m², une ligne par année, une colonne par zone."""
    return df.pivot_table(index="annee", columns="zone", values="prix_m2", aggfunc="median").round(0)


def prix_par_tranche(df: pd.DataFrame) -> pd.DataFrame:
    """Effectif, prix moyen et médian par zone et tranche de la note de cadrage."""
    tranche = pd.cut(df["dist_station_m"], TRANCHES_NOTE, right=False, labels=LABELS_NOTE)
    return (df.groupby(["zone", tranche], observed=True)["prix_m2"]
              .agg(n="size", moyenne="mean", mediane="median")
              .round(0).astype({"n": int}))


def prix_courbe_fine(df: pd.DataFrame, n_min: int = 30) -> pd.DataFrame:
    """Prix médian par tranche de 50 m (jusqu'à 1 000 m puis tranches plus larges)."""
    bornes = list(range(0, 1001, 50)) + [1250, 1500, 2000, np.inf]
    tranche = pd.cut(df["dist_station_m"], bornes, right=False)
    g = (df.groupby(["zone", tranche], observed=True)["prix_m2"]
           .agg(n="size", mediane="median").reset_index(names=["zone", "tranche"]))
    droite = g["tranche"].astype(object).map(lambda i: float(i.right))
    gauche = g["tranche"].astype(object).map(lambda i: float(i.left))
    g = g.assign(distance_m=(gauche + droite) / 2)
    return g[(g["n"] >= n_min) & np.isfinite(droite)].drop(columns="tranche")


# --------------------------------------------------------------------------- tranches de distance

BORNES_100 = [0, 100, 200, 300, 400, 600, 800, 1000, 1500, np.inf]
BORNES_50 = list(range(0, 1001, 50)) + [1500, 2000, np.inf]
BORNES_COMMUNE = list(range(0, 1001, 50)) + [np.inf]


def tranche(distance: pd.Series, bornes: list) -> pd.Series:
    """Découpe une distance en tranches nommées « 0-100 », « 100-200 », ..., « 1500+ »."""
    noms = [f"{int(a)}-{int(b)}" if np.isfinite(b) else f"{int(a)}+" for a, b in zip(bornes[:-1], bornes[1:])]
    return pd.cut(distance, bornes, labels=noms, right=False, include_lowest=True)


def prix_par_tranche_fine(df: pd.DataFrame, bornes: list, par_zone: bool = True) -> pd.DataFrame:
    """Effectif, prix médian et prix moyen par tranche de distance (et par zone)."""
    cles = (["zone"] if par_zone else []) + [tranche(df["dist_station_m"], bornes).rename("tranche")]
    return (df.groupby(cles, observed=True)["prix_m2"]
              .agg(nombre_transactions="size", prix_median="median", prix_moyen="mean")
              .reset_index())


def evolution_annuelle(df: pd.DataFrame, bornes: list = BORNES_100) -> pd.DataFrame:
    """Prix médian par zone, année et tranche, et évolution en % par rapport à 2021."""
    t = (df.assign(tranche=tranche(df["dist_station_m"], bornes))
           .groupby(["zone", "annee", "tranche"], observed=True)["prix_m2"]
           .agg(nombre_transactions="size", prix_median="median").reset_index())
    base = t[t["annee"] == t["annee"].min()][["zone", "tranche", "prix_median"]]
    t = t.merge(base.rename(columns={"prix_median": "prix_2021"}), on=["zone", "tranche"], how="left")
    t["evolution_depuis_2021_pct"] = (t["prix_median"] / t["prix_2021"] - 1) * 100
    return t


# --------------------------------------------------------------------------- comparaison à l'intérieur des communes

def ecarts_communes(df: pd.DataFrame, ref=(200, 400)) -> pd.DataFrame:
    """Pour chaque commune et chaque tranche de 50 m : prix médian, et écart à la tranche de référence (en %)."""
    t = (df.assign(tranche=tranche(df["dist_station_m"], BORNES_COMMUNE))
           .groupby(["zone", "commune_id", "nom_commune", "tranche"], observed=True)["prix_m2"]
           .agg(nombre_transactions="size", prix_median="median").reset_index())
    en_ref = df[df["dist_station_m"].between(*ref, inclusive="left")]
    r = (en_ref.groupby("commune_id")["prix_m2"]
               .agg(prix_reference="median", n_reference="size").reset_index())
    t = t.merge(r, on="commune_id", how="left")
    t["ecart_relatif_pct"] = (t["prix_median"] / t["prix_reference"] - 1) * 100
    return t


def synthese_ecarts(stats: pd.DataFrame, seuil: int = 30) -> pd.DataFrame:
    """Résume les écarts par zone et tranche, en ne gardant que les groupes d'au moins `seuil` ventes."""
    ok = stats[(stats["nombre_transactions"] >= seuil) & (stats["n_reference"] >= seuil)
               & (stats["prix_reference"] > 0)]
    out = (ok.groupby(["zone", "tranche"], observed=True)["ecart_relatif_pct"]
             .agg(nombre_communes="size", ecart_median_pct="median", ecart_moyen_pct="mean",
                  part_communes_positives=lambda x: (x > 0).mean() * 100).reset_index())
    out["seuil_effectif"] = seuil
    return out


def robustesse_seuils(stats: pd.DataFrame, seuils=(20, 30, 50, 100)) -> pd.DataFrame:
    """Synthèse des écarts pour plusieurs seuils d'effectif minimal."""
    return pd.concat([synthese_ecarts(stats, s) for s in seuils], ignore_index=True)


def detecter_intervalles(synthese: pd.DataFrame, seuil: int) -> pd.DataFrame:
    """Séquences d'au moins 2 tranches consécutives où l'écart médian est positif
    et où plus de la moitié des communes ont un écart positif."""
    colonnes = ["zone", "seuil_effectif", "debut_m", "fin_m", "n_tranches",
                "ecart_median_moyen_pct", "part_positive_moyenne_pct"]
    lignes = []
    for zone, n_min in [("Paris", 10), ("Petite couronne", 20)]:
        t = synthese[(synthese["zone"] == zone) & (synthese["seuil_effectif"] == seuil)
                     & (synthese["nombre_communes"] >= n_min) & (synthese["ecart_median_pct"] > 0)
                     & (synthese["part_communes_positives"] > 50)].copy()
        t["tranche"] = t["tranche"].astype(str)
        t = t[t["tranche"].str.match(r"^\d+-\d+$")]
        if t.empty:
            continue
        t["debut"] = t["tranche"].str.extract(r"^(\d+)-")[0].astype(int)
        t["fin"] = t["tranche"].str.extract(r"-(\d+)$")[0].astype(int)
        t = t.sort_values("debut")
        t["sequence"] = (t["debut"] != t["fin"].shift()).cumsum()   # nouvelle séquence si trou
        for _, g in t.groupby("sequence"):
            if len(g) >= 2:
                lignes.append([zone, seuil, g["debut"].min(), g["fin"].max(), len(g),
                               g["ecart_median_pct"].mean(), g["part_communes_positives"].mean()])
    return pd.DataFrame(lignes, columns=colonnes)


# --------------------------------------------------------------------------- contrôles et résumés

COLONNES_REQUISES = ["annee", "code_departement", "code_commune", "nom_commune",
                     "valeur_fonciere", "surface_reelle_bati", "prix_m2", "dist_station_m"]


def controler_donnees(df: pd.DataFrame) -> pd.DataFrame:
    """Vérifie les colonnes indispensables et écarte les lignes sans prix ni distance exploitable."""
    manquantes = [c for c in COLONNES_REQUISES if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes indispensables absentes : {manquantes}")
    ok = df[df["prix_m2"].notna() & df["dist_station_m"].notna()
            & (df["prix_m2"] > 0) & (df["dist_station_m"] >= 0)].copy()
    print(f"{len(ok):,} observations retenues sur {len(df):,}".replace(",", " "))
    return ok


def valeurs_manquantes(df: pd.DataFrame) -> pd.DataFrame:
    """Nombre et part de valeurs manquantes par colonne (colonnes concernées seulement)."""
    t = pd.DataFrame({"manquantes": df.isna().sum(), "part_%": df.isna().mean() * 100})
    return t[t["manquantes"] > 0].sort_values("part_%", ascending=False).round(2)


def description_zone(df: pd.DataFrame) -> pd.DataFrame:
    """Effectif, prix et distance à la station par zone."""
    return (df.groupby("zone")
              .agg(nombre_transactions=("prix_m2", "size"), prix_median=("prix_m2", "median"),
                   prix_moyen=("prix_m2", "mean"), ecart_type=("prix_m2", "std"),
                   distance_mediane=("dist_station_m", "median"),
                   distance_moyenne=("dist_station_m", "mean"))
              .reset_index())


def meilleures_tranches(table: pd.DataFrame, n_min: int = 100, n: int = 5) -> pd.DataFrame:
    """Les `n` tranches au prix médian le plus élevé de chaque zone (groupes d'au moins `n_min` ventes)."""
    t = table[table["nombre_transactions"] >= n_min]
    return t.sort_values("prix_median", ascending=False).groupby("zone").head(n)


# --------------------------------------------------------------------------- métro contre RER

BORNES_MODES = [0, 200, 400, 600, 800, 1000, 1500, np.inf]


def extraire_mode(valeur) -> str | float:
    """Mode d'une station à partir de `modes_station` : METRO, RER, TRAM, TRAIN, MULTI_MODE ou AUTRE."""
    if pd.isna(valeur):
        return np.nan
    s = str(valeur).upper()
    presents = [m for m, mots in [("METRO", ("METRO", "MÉTRO")), ("RER", ("RER",)), ("TRAM", ("TRAM",)),
                                  ("TRAIN", ("TRAIN", "TRANSILIEN"))] if any(w in s for w in mots)]
    if len(presents) == 1:
        return presents[0]
    return "MULTI_MODE" if presents else "AUTRE"


def comparaison_modes(df: pd.DataFrame, n_min: int = 30):
    """Prix par tranche de distance pour les stations de métro et de RER (hors stations multi-modes).

    Renvoie (resume, pentes, courbe) : tableau par tranche avec écart RER - métro,
    écart de médiane et corrélation de rang pour chaque mode, et données de la courbe.
    """
    d = df.assign(mode_station=df["modes_station"].map(extraire_mode))
    d = d[d["mode_station"].isin(["METRO", "RER"]) & d["dist_station_m"].notna() & d["prix_m2"].notna()].copy()
    d["tranche"] = tranche(d["dist_station_m"], BORNES_MODES)

    g = (d.groupby(["tranche", "mode_station"], observed=True)["prix_m2"]
           .agg(n="size", mediane="median", moyenne="mean").reset_index())
    resume = g.pivot(index="tranche", columns="mode_station", values=["n", "mediane", "moyenne"])
    resume.columns = [f"{stat}_{mode}" for stat, mode in resume.columns]
    resume["ecart_mediane_RER_moins_METRO"] = resume["mediane_RER"] - resume["mediane_METRO"]
    resume["ratio_mediane_RER_sur_METRO"] = resume["mediane_RER"] / resume["mediane_METRO"]

    lignes = []
    for mode, sub in d.groupby("mode_station"):
        proche = sub.loc[sub["tranche"] == "0-200", "prix_m2"].median()
        loin = sub.loc[sub["tranche"] == "600-800", "prix_m2"].median()
        lignes.append({"mode": mode, "n": len(sub), "mediane_0-200": proche, "mediane_600-800": loin,
                       "ecart_%": (loin / proche - 1) * 100,
                       "correlation_spearman": sub["dist_station_m"].corr(sub["prix_m2"], method="spearman")})
    return resume.round(2), pd.DataFrame(lignes).round(2), g[g["n"] >= n_min]


# --------------------------------------------------------------------------- effet de la distance, sans modèle

BORNES_REF = [0, 100, 200, 400, 600, 1000]          # tranches de comparaison à la référence
REFERENCE = "200-400"
CLASSES_SURFACE = [0, 30, 50, 80, np.inf]            # studios, 2 pièces, 3 pièces, grands logements


def preparer(df: pd.DataFrame, queue: float = 0.01) -> pd.DataFrame:
    """Écarte les 1 % de prix les plus bas et les plus hauts de chaque zone, ajoute prix relatif et classe de surface."""
    d = df.dropna(subset=["prix_m2", "surface_reelle_bati", "dist_station_m", "commune_id"]).copy()
    q = d.groupby("zone")["prix_m2"].transform
    d = d[(d["prix_m2"] >= q(lambda s: s.quantile(queue))) & (d["prix_m2"] <= q(lambda s: s.quantile(1 - queue)))].copy()
    # prix relatif : écart (%) au prix médian de la même commune la même année
    mediane = d.groupby(["commune_id", "annee"])["prix_m2"].transform("median")
    d["prix_relatif_pct"] = (d["prix_m2"] / mediane - 1) * 100
    d["classe_surface"] = pd.cut(d["surface_reelle_bati"], CLASSES_SURFACE, right=False)
    return d


def correlations(d: pd.DataFrame, par: str | None = None, dist_max: float | None = None) -> pd.DataFrame:
    """Corrélation entre la distance à la station et le prix (brut, puis relatif à la commune).

    Spearman (rangs) ne dépend pas des valeurs extrêmes ; Pearson mesure la liaison linéaire.
    """
    if dist_max is not None:
        d = d[d["dist_station_m"] <= dist_max]
    groupes = [("Ensemble", d)] + [(z, d[d["zone"] == z]) for z in ZONES]
    lignes = []
    for nom, g in groupes:
        sous = [(None, g)] if par is None else list(g.groupby(par))
        for cle, h in sous:
            ligne = {"zone": nom, "n": len(h)}
            if par is not None:
                ligne[par] = cle
            for var, label in [("prix_m2", "prix"), ("prix_relatif_pct", "prix_relatif")]:
                for meth in ("spearman", "pearson"):
                    ligne[f"{meth}_{label}"] = h["dist_station_m"].corr(h[var], method=meth)
            lignes.append(ligne)
    return pd.DataFrame(lignes).round(3)


def variation_par_tranche(d: pd.DataFrame, largeur: int = 200, dist_max: int = 1600, n_min: int = 30) -> pd.DataFrame:
    """Prix médian (brut et relatif à la commune) par tranche de `largeur` m, et variation d'une tranche à la suivante.

    `variation_par_100m` = différence de médianes entre deux tranches consécutives, ramenée à 100 m.
    """
    bornes = list(range(0, dist_max + 1, largeur)) + [np.inf]
    t = (d.assign(tranche=tranche(d["dist_station_m"], bornes))
           .groupby(["zone", "tranche"], observed=True)
           .agg(n=("prix_m2", "size"), prix_median=("prix_m2", "median"),
                prix_relatif_median_pct=("prix_relatif_pct", "median")).reset_index())
    t = t[t["n"] >= n_min].copy()
    t["tranche"] = t["tranche"].astype(str)
    out = []
    for zone, g in t.groupby("zone", sort=False):
        g = g.copy()
        g["variation_eur"] = g["prix_median"].diff()
        g["variation_pct"] = g["prix_median"].pct_change() * 100
        g["variation_par_100m_eur"] = g["variation_eur"] / (largeur / 100)
        g["variation_relatif_par_100m_pts"] = g["prix_relatif_median_pct"].diff() / (largeur / 100)
        out.append(g)
    return pd.concat(out, ignore_index=True).round(2)


def _moyenne_ponderee(g: pd.DataFrame, col: str, poids: str) -> float:
    return float(np.average(g[col], weights=g[poids]))


def ecarts_stratifies(d: pd.DataFrame, ref: tuple = (200, 400), n_min: int = 10) -> pd.DataFrame:
    """Écart de prix de chaque tranche par rapport à la tranche de référence, à biens comparables.

    On compare uniquement des ventes de même commune, de même année et de même classe de surface
    (une « cellule »). Dans chaque cellule, écart = médiane de la tranche / médiane de la référence - 1.
    Les écarts sont ensuite moyennés sur les cellules, pondérés par le nombre de ventes de la tranche.
    """
    cles = ["zone", "commune_id", "annee", "classe_surface"]
    d = d.assign(tranche=pd.cut(d["dist_station_m"], BORNES_REF, right=False,
                                labels=["0-100", "100-200", "200-400", "400-600", "600-1000"]).astype(str))
    d = d[d["tranche"] != "nan"]
    t = d.groupby(cles + ["tranche"], observed=True)["prix_m2"].agg(n="size", mediane="median").reset_index()
    r = t[t["tranche"] == REFERENCE].rename(columns={"n": "n_ref", "mediane": "mediane_ref"}).drop(columns="tranche")
    t = t.merge(r, on=cles)
    t = t[(t["n"] >= n_min) & (t["n_ref"] >= n_min) & (t["tranche"] != REFERENCE)]
    t["ecart_pct"] = (t["mediane"] / t["mediane_ref"] - 1) * 100
    res = (t.groupby(["zone", "tranche"], observed=True)[["ecart_pct", "n"]]
             .apply(lambda g: pd.Series({"ecart_pct": _moyenne_ponderee(g, "ecart_pct", "n"),
                                         "ecart_median_cellules_pct": g["ecart_pct"].median(),
                                         "part_cellules_positives_%": (g["ecart_pct"] > 0).mean() * 100,
                                         "n_cellules": len(g), "n_ventes": int(g["n"].sum())}))
             .reset_index())
    res["tranche"] = pd.Categorical(res["tranche"], ["0-100", "100-200", "400-600", "600-1000"], ordered=True)
    return res.sort_values(["zone", "tranche"]).round(2).reset_index(drop=True)


def prime_stratifiee(d: pd.DataFrame, proche: float = 300, loin: float = 600, n_min: int = 10,
                     par_annee: bool = False) -> pd.DataFrame:
    """Prime des ventes à moins de `proche` m par rapport à celles à plus de `loin` m, à biens comparables.

    Même principe que `ecarts_stratifies` : médiane proche / médiane loin - 1 dans chaque cellule
    (commune, année, classe de surface), moyennée avec pour poids le plus petit des deux effectifs.
    """
    cles = ["zone", "commune_id", "annee", "classe_surface"]
    x = d.assign(groupe=np.select([d["dist_station_m"] < proche, d["dist_station_m"] >= loin], ["proche", "loin"], "autre"))
    x = x[x["groupe"] != "autre"]
    t = x.groupby(cles + ["groupe"], observed=True)["prix_m2"].agg(n="size", mediane="median").unstack("groupe").dropna()
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    t = t.reset_index()
    t = t[(t["n_proche"] >= n_min) & (t["n_loin"] >= n_min)].copy()
    t["prime_pct"] = (t["mediane_proche"] / t["mediane_loin"] - 1) * 100
    t["poids"] = t[["n_proche", "n_loin"]].min(axis=1)
    groupes = ["zone", "annee"] if par_annee else ["zone"]
    res = (t.groupby(groupes)[["prime_pct", "poids"]]
             .apply(lambda g: pd.Series({"prime_%": _moyenne_ponderee(g, "prime_pct", "poids"),
                                         "mediane_cellules_%": g["prime_pct"].median(),
                                         "q25_%": g["prime_pct"].quantile(.25), "q75_%": g["prime_pct"].quantile(.75),
                                         "part_cellules_positives_%": (g["prime_pct"] > 0).mean() * 100,
                                         "n_cellules": len(g)}))
             .reset_index())
    return res.round(2)


def robustesse_prime(d_brut: pd.DataFrame) -> pd.DataFrame:
    """Prime proche/loin stratifiée pour plusieurs seuils de proximité et d'effectif minimal."""
    d = preparer(d_brut)
    essais = [("référence : <300 m vs ≥600 m, 10 ventes min.", d, 300, 600, 10),
              ("<200 m vs ≥500 m", d, 200, 500, 10), ("<400 m vs ≥800 m", d, 400, 800, 10),
              ("20 ventes min. par groupe", d, 300, 600, 20), ("5 ventes min. par groupe", d, 300, 600, 5),
              ("sans écarter les prix extrêmes", d_brut.assign(
                  prix_relatif_pct=0, classe_surface=pd.cut(d_brut["surface_reelle_bati"], CLASSES_SURFACE, right=False)),
               300, 600, 10)]
    blocs = []
    for nom, echantillon, p, l, n in essais:
        r = prime_stratifiee(echantillon, p, l, n)
        r.insert(0, "specification", nom)
        blocs.append(r)
    return pd.concat(blocs, ignore_index=True)


# --------------------------------------------------------------------------- prolongements des lignes 11 et 14

LIGNES_PROLONGEES = ["METRO 11", "METRO 14"]
OUVERTURE_DEBUT, OUVERTURE_FIN = pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-30")
DESSERVI, TEMOIN = "Desservi en 2024", "Témoin"


def distances_prolongements(df: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque vente : distance à la plus proche des stations ouvertes en juin 2024 sur les lignes 11 et 14
    (`dist_nouvelle_m`, `nouvelle_station`) et distance à la plus proche station du réseau d'avant
    l'ouverture (`dist_avant_m`), quelle que soit la date de la vente.

    Ces distances décrivent la position du logement, pas l'état du réseau à la date de vente : elles
    servent à suivre les mêmes quartiers avant et après l'ouverture.
    """
    import geopandas as gpd

    from . import stations as S

    lignes = S.read_stations_lignes()
    est_nouvelle = (lignes["ligne"].isin(LIGNES_PROLONGEES)
                    & lignes["date_mise"].between(OUVERTURE_DEBUT, OUVERTURE_FIN))
    nouvelles = lignes.loc[est_nouvelle, ["nom_station", "ligne", "geometry"]].rename(
        columns={"nom_station": "nouvelle_station", "ligne": "ligne_nouvelle"})
    if nouvelles.empty:
        raise ValueError("Aucune station des lignes 11 ou 14 ouverte en juin 2024 dans le fichier IDFM.")
    avant = S.stations_ouvertes(lignes, OUVERTURE_DEBUT - pd.Timedelta(days=1))[["geometry"]]

    d = df.dropna(subset=["longitude", "latitude", "prix_m2"]).reset_index(drop=True)
    ventes = S.dvf_to_gdf(d)[["geometry"]]
    proche = gpd.sjoin_nearest(ventes, nouvelles, how="left", distance_col="dist_nouvelle_m")
    proche = proche[~proche.index.duplicated(keep="first")]
    ancien = gpd.sjoin_nearest(ventes, avant, how="left", distance_col="dist_avant_m")
    ancien = ancien[~ancien.index.duplicated(keep="first")]

    d["nouvelle_station"] = proche["nouvelle_station"]
    d["ligne_nouvelle"] = proche["ligne_nouvelle"]
    d["dist_nouvelle_m"] = proche["dist_nouvelle_m"]
    d["dist_avant_m"] = ancien["dist_avant_m"]
    return d


def groupes_prolongements(d_prol: pd.DataFrame, rayon: float = 600, sans_station_avant: float = 800,
                          temoin_au_dela: float = 1200) -> pd.DataFrame:
    """Ventes des groupes « desservi » et « témoin » (colonne `groupe`).

    - desservi : à moins de `rayon` m d'une nouvelle station et à plus de `sans_station_avant` m
      de toute station existant avant l'ouverture ;
    - témoin : ventes des mêmes communes situées à plus de `temoin_au_dela` m de toute nouvelle station.
    """
    desservi = (d_prol["dist_nouvelle_m"] < rayon) & (d_prol["dist_avant_m"] > sans_station_avant)
    communes = d_prol.loc[desservi, "commune_id"].unique()
    temoin = d_prol["commune_id"].isin(communes) & (d_prol["dist_nouvelle_m"] > temoin_au_dela)
    g = d_prol[desservi | temoin].copy()
    g["groupe"] = np.where(desservi[desservi | temoin], DESSERVI, TEMOIN)
    return g


def evolution_prolongements(g_prol: pd.DataFrame) -> pd.DataFrame:
    """Par année et par groupe : nombre de ventes, prix médian au m² et indice (base 100 la première année).

    `ecart_indices` = indice du groupe desservi - indice du groupe témoin (en points).
    """
    t = g_prol.pivot_table(index="annee", columns="groupe", values="prix_m2",
                           aggfunc=["size", "median"]).sort_index()
    out = pd.DataFrame(index=t.index)
    for grp in (DESSERVI, TEMOIN):
        out[f"n {grp}"] = t[("size", grp)].astype(int)
        out[f"prix_median {grp}"] = t[("median", grp)].round(0)
        out[f"indice {grp}"] = (100 * t[("median", grp)] / t[("median", grp)].iloc[0]).round(1)
    out["ecart_indices"] = (out[f"indice {DESSERVI}"] - out[f"indice {TEMOIN}"]).round(1)
    out.index.name = "annee"
    return out

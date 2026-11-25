"""
Brique 6 : régression hédonique + intervalles de confiance sur la prime de proximité.

Prérequis : pip install statsmodels
Sorties   : outputs/figures/04_prime_hedonique_ic.png
            outputs/tables/prime_hedonique_par_annee.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------------------------------------------------
# 0. Paramètres
# ----------------------------------------------------------------------
RACINE = Path("/home/onyxia/work/Projet_Data_2A_S1")
PROCHE, LOIN = 300, 600      # mêmes seuils que la brique 5
N_MIN = 20                   # effectif min par groupe et par commune (bootstrap)
B = 200                      # nombre de tirages bootstrap
DIST_MAX_CONTINU = 1000      # la régression « distance continue » se limite à ≤ 1000 m
rng = np.random.default_rng(42)

# ----------------------------------------------------------------------
# 1. Chargement et préparation
# ----------------------------------------------------------------------
df = pd.read_parquet(RACINE / "data/processed/dvf_appartements_stations.parquet")
df["annee"] = df["date_mutation"].dt.year
df["zone"] = np.where(df["code_departement"].astype(str).str.strip() == "75",
                      "Paris", "Petite couronne")


def trouve(candidats, obligatoire=True):
    """Renvoie le premier nom de colonne présent parmi les candidats."""
    for c in candidats:
        if c in df.columns:
            return c
    if obligatoire:
        raise KeyError(f"Aucune de ces colonnes n'existe : {candidats}\n"
                       f"Colonnes disponibles : {list(df.columns)}")
    return None


# >>> Si tes colonnes ont d'autres noms, ajoute-les dans ces listes <<<
col_surface = trouve(["surface_reelle_bati", "surface", "surface_m2", "surface_bati"])
col_pieces = trouve(["nombre_pieces_principales", "nb_pieces", "nombre_pieces",
                     "n_pieces"], obligatoire=False)
# Effet fixe de lieu : code_commune (donne les arrondissements à Paris : 75101...)
col_lieu = trouve(["code_commune", "code_insee", "nom_commune"])

df["surface"] = df[col_surface].astype(float)
df["lieu"] = df[col_lieu].astype(str)
if col_pieces:
    df["pieces"] = df[col_pieces].clip(upper=5).fillna(-1).astype(int)  # 5 = « 5 et + »
# Dépendances (cave, parking...) : leur prix est inclus dans la valeur foncière
col_dep = trouve(["n_dependances"], obligatoire=False)
if col_dep:
    df["n_dep"] = df[col_dep].fillna(0).clip(upper=3).astype(int)

print(f"Surface : {col_surface} | Pièces : {col_pieces} | Effet fixe lieu : {col_lieu}")
print("Nombre de lieux distincts par zone (Paris doit être ~20) :")
print(df.groupby("zone")["lieu"].nunique(), "\n")

# Nettoyage : valeurs manquantes, surfaces aberrantes, 1 % de chaque queue de prix par zone
d = df.dropna(subset=["prix_m2", "surface", "dist_station_m", "lieu", "annee"])
d = d[(d["surface"] >= 9) & (d["surface"] <= 300) & (d["prix_m2"] > 0)].copy()
q = d.groupby("zone")["prix_m2"].transform
d = d[(d["prix_m2"] >= q(lambda s: s.quantile(0.01))) &
      (d["prix_m2"] <= q(lambda s: s.quantile(0.99)))].copy()
print(f"{len(d)} ventes conservées sur {len(df)}\n")

d["dist_100"] = d["dist_station_m"] / 100
d["groupe"] = np.select([d["dist_station_m"] < PROCHE, d["dist_station_m"] >= LOIN],
                        ["proche", "loin"], default="milieu")

controles = ("np.log(surface) + C(lieu)" + (" + C(pieces)" if col_pieces else "")
             + (" + C(n_dep)" if col_dep else ""))


def ajuste(formule, data):
    """MCO avec écarts-types robustes, regroupés par commune/arrondissement."""
    groupes = data["lieu"].astype("category").cat.codes.values
    return smf.ols(formule, data=data).fit(cov_type="cluster", cov_kwds={"groups": groupes})


# ----------------------------------------------------------------------
# 2. Régression hédonique : effet de la distance (continue)
# ----------------------------------------------------------------------
print("=" * 70)
print("MODÈLE 1 : log(prix_m2) ~ distance (par 100 m) + contrôles")
print("=" * 70)
for zone in ["Paris", "Petite couronne"]:
    dz = d[(d["zone"] == zone) & (d["dist_station_m"] <= DIST_MAX_CONTINU)]
    m = ajuste(f"np.log(prix_m2) ~ dist_100 + C(annee) + {controles}", dz)
    b = m.params["dist_100"]
    lo, hi = m.conf_int().loc["dist_100"]
    print(f"{zone:16s} : +100 m de distance  ->  {100*(np.exp(b)-1):+.2f} % "
          f"[IC95 % : {100*(np.exp(lo)-1):+.2f} ; {100*(np.exp(hi)-1):+.2f}]  "
          f"(n={int(m.nobs)}, R²={m.rsquared:.2f})")

# ----------------------------------------------------------------------
# 3. Prime proche/loin, par année, avec contrôles (IC analytiques)
# ----------------------------------------------------------------------
dd = d[d["groupe"] != "milieu"].copy()
dd["proche"] = (dd["groupe"] == "proche").astype(int)

lignes = []
for zone in ["Paris", "Petite couronne"]:
    dz = dd[dd["zone"] == zone]

    # a) prime moyenne sur toute la période
    m_tot = ajuste(f"np.log(prix_m2) ~ proche + C(annee) + {controles}", dz)
    b = m_tot.params["proche"]; lo, hi = m_tot.conf_int().loc["proche"]
    print(f"\n{zone} : prime moyenne 2021-2025 (proche < {PROCHE} m vs ≥ {LOIN} m), "
          f"à caractéristiques égales : {100*(np.exp(b)-1):+.1f} % "
          f"[{100*(np.exp(lo)-1):+.1f} ; {100*(np.exp(hi)-1):+.1f}]")

    # b) prime année par année (interaction proche x année)
    m_an = ajuste(f"np.log(prix_m2) ~ C(annee) + proche:C(annee) + {controles}", dz)
    ic = m_an.conf_int()
    for annee in sorted(dz["annee"].unique()):
        nom = [p for p in m_an.params.index if p.startswith("proche:") and str(annee) in p][0]
        lignes.append({
            "zone": zone, "annee": annee,
            "prime_hedonique_pct": 100 * (np.exp(m_an.params[nom]) - 1),
            "ic_bas": 100 * (np.exp(ic.loc[nom, 0]) - 1),
            "ic_haut": 100 * (np.exp(ic.loc[nom, 1]) - 1),
            "n": int((dz["annee"] == annee).sum()),
        })
hed = pd.DataFrame(lignes)

# ----------------------------------------------------------------------
# 4. Bootstrap de la prime « à commune égale » (brique 5) pour avoir un IC
# ----------------------------------------------------------------------
def prime_commune_egale(x):
    g = x.groupby(["lieu", "groupe"])["prix_m2"].agg(["median", "size"]).unstack("groupe")
    g = g.dropna()
    if g.empty:
        return np.nan
    n_p, n_l = g[("size", "proche")], g[("size", "loin")]
    ok = (n_p >= N_MIN) & (n_l >= N_MIN)
    if ok.sum() == 0:
        return np.nan
    prime = 100 * (g[("median", "proche")][ok] / g[("median", "loin")][ok] - 1)
    return np.average(prime, weights=np.minimum(n_p[ok], n_l[ok]))


boot = []
for (zone, annee), x in dd.groupby(["zone", "annee"]):
    pt = prime_commune_egale(x)
    tirages = [prime_commune_egale(x.sample(frac=1, replace=True,
                                            random_state=int(rng.integers(1e9))))
               for _ in range(B)]
    tirages = np.array(tirages, dtype=float)
    boot.append({"zone": zone, "annee": annee, "prime_brique5_pct": pt,
                 "b5_ic_bas": np.nanpercentile(tirages, 2.5),
                 "b5_ic_haut": np.nanpercentile(tirages, 97.5)})
boot = pd.DataFrame(boot)

res = hed.merge(boot, on=["zone", "annee"])
pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
print("\nRésultats par année :")
print(res.round(1).to_string(index=False))

dossier_tab = RACINE / "outputs/tables"; dossier_tab.mkdir(parents=True, exist_ok=True)
dossier_fig = RACINE / "outputs/figures"; dossier_fig.mkdir(parents=True, exist_ok=True)
res.round(2).to_csv(dossier_tab / "prime_hedonique_par_annee.csv", index=False)

# ----------------------------------------------------------------------
# 5. Graphique
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, zone in zip(axes, ["Paris", "Petite couronne"]):
    r = res[res["zone"] == zone]
    ax.fill_between(r["annee"], r["ic_bas"], r["ic_haut"], color="#2a78d6", alpha=0.18)
    ax.plot(r["annee"], r["prime_hedonique_pct"], color="#2a78d6", lw=2, marker="o",
            label="Régression (surface, pièces, commune)")
    ax.fill_between(r["annee"], r["b5_ic_bas"], r["b5_ic_haut"], color="#eb6834", alpha=0.18)
    ax.plot(r["annee"], r["prime_brique5_pct"], color="#eb6834", lw=2, marker="o",
            label="À commune égale (brique 5)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(sorted(res["annee"].unique()))
    ax.set_title(zone, loc="left")
    ax.set_ylabel(f"Prime < {PROCHE} m vs ≥ {LOIN} m (%)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
fig.suptitle("Prime de proximité avec intervalles de confiance à 95 %", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(dossier_fig / "04_prime_hedonique_ic.png", dpi=150)
print("\nFigure enregistrée dans", dossier_fig)
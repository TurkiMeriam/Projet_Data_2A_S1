import numpy as np
import pandas as pd

df = pd.read_parquet("/home/onyxia/work/Projet_Data_2A_S1/data/processed/dvf_appartements_stations.parquet")

df["annee"] = df["date_mutation"].dt.year
df["zone"] = np.where(df["code_departement"] == 75, "Paris", "Petite couronne")

bornes = list(range(0, 1001, 50)) + [1250, 1500, 2000, np.inf]
df["tranche"] = pd.cut(df["dist_station_m"], bornes, right=False)

print(df.groupby(["zone", "tranche"], observed=True).size())
# Brique 2 : prix au m² par zone, année et tranche de distance
table = (
    df.groupby(["zone", "annee", "tranche"], observed=True)["prix_m2"]
      .agg(n="size", moyenne="mean", mediane="median")
      .reset_index()
)

# Une tranche avec trop peu de ventes donne une moyenne peu fiable
table = table[table["n"] >= 30]

# Vue lisible : une ligne par tranche, une colonne par année (prix médian)
pivot = table.pivot_table(index=["zone", "tranche"], columns="annee",
                          values="mediane", observed=True).round(0)

pd.set_option("display.max_rows", 100)
print(pivot)
# Brique 3 : graphique prix médian selon la distance, une courbe par année
import matplotlib
matplotlib.use("Agg")          # pas d'écran : on enregistre l'image dans un fichier
import matplotlib.pyplot as plt
from pathlib import Path

# Pour l'axe des x, on place chaque tranche à son milieu (ex. [100, 150) -> 125 m)
def milieu(intervalle):
    if np.isinf(intervalle.right):
        return intervalle.left + 250      # dernière tranche ouverte « 2000 m et + »
    return (intervalle.left + intervalle.right) / 2

table["distance_m"] = table["tranche"].apply(milieu).astype(float)

# Années en dégradé de bleu : du clair (2021) au foncé (2025)
couleurs = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, zone in zip(axes, ["Paris", "Petite couronne"]):
    tz = table[(table["zone"] == zone) & (table["distance_m"] <= 2000)]
    for couleur, (annee, g) in zip(couleurs, tz.groupby("annee")):
        g = g.sort_values("distance_m")
        ax.plot(g["distance_m"], g["mediane"], color=couleur, lw=2, label=str(annee))
    ax.set_title(zone, loc="left")
    ax.set_xlabel("Distance à la station la plus proche (m)")
    ax.set_ylabel("Prix médian (€/m²)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Année", frameon=False)

fig.suptitle("Prix au m² selon la distance au métro/RER, par année", x=0.01, ha="left")
fig.tight_layout()

dossier = Path("/home/onyxia/work/Projet_Data_2A_S1/outputs/figures")
dossier.mkdir(parents=True, exist_ok=True)
fig.savefig(dossier / "01_prix_selon_distance_par_annee.png", dpi=150)
print("Figure enregistrée dans", dossier)

#nouveau_graphe

# === Brique 4 : indice base 100 par classe de distance ===

# Classes plus larges : assez de ventes par année pour être fiables
df["classe"] = pd.cut(df["dist_station_m"], [0, 200, 400, 600, np.inf], right=False,
                      labels=["0-200 m", "200-400 m", "400-600 m", "600 m et +"])

ind = (df.groupby(["zone", "classe", "annee"], observed=True)["prix_m2"]
         .median().rename("mediane").reset_index()
         .sort_values(["zone", "classe", "annee"]))

# Indice : prix de l'année / prix de 2021 de la même classe, x 100
base_2021 = ind[ind["annee"] == 2021].set_index(["zone", "classe"])["mediane"]
ind["indice"] = 100 * ind["mediane"] / ind.set_index(["zone", "classe"]).index.map(base_2021)

# Variation d'une année sur l'autre (%)
ind["var_pct"] = 100 * ind.groupby(["zone", "classe"], observed=True)["mediane"].pct_change()

# Retournement : la variation change de signe par rapport à l'année d'avant
signe = np.sign(ind["var_pct"])
signe_avant = signe.groupby([ind["zone"], ind["classe"]], observed=True).shift()
ind["retournement"] = (signe * signe_avant) < 0

print(ind.pivot_table(index=["zone", "classe"], columns="annee",
                      values="indice", observed=True).round(1))
print(ind.pivot_table(index=["zone", "classe"], columns="annee",
                      values="var_pct", observed=True).round(1))
print(ind[ind["retournement"]][["zone", "classe", "annee", "var_pct"]].round(1))

# Graphique : une courbe par classe, du foncé (proche) au clair (loin)
couleurs_classes = ["#0d366b", "#256abf", "#5598e7", "#9ec5f4"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, zone in zip(axes, ["Paris", "Petite couronne"]):
    iz = ind[ind["zone"] == zone]
    for couleur, (classe, g) in zip(couleurs_classes, iz.groupby("classe", observed=True)):
        ax.plot(g["annee"], g["indice"], color=couleur, lw=2, marker="o", label=classe)
    ax.axhline(100, color="grey", lw=0.8)
    ax.set_xticks([2021, 2022, 2023, 2024, 2025])
    ax.set_title(zone, loc="left")
    ax.set_ylabel("Indice prix médian (base 100 = 2021)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Distance", frameon=False)
fig.suptitle("Évolution des prix selon la distance à la station", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(dossier / "02_indice_base100.png", dpi=150)

"""figure 3"""
# === Brique 5 : prime de proximité à commune égale ===

PROCHE = 300   # « proche » : moins de 300 m d'une station
LOIN = 600     # « loin » : 600 m et plus
N_MIN = 20     # on garde une commune seulement si elle a au moins 20 ventes de chaque groupe

df["groupe"] = np.select(
    [df["dist_station_m"] < PROCHE, df["dist_station_m"] >= LOIN],
    ["proche", "loin"],
    default="milieu",
)
d = df[df["groupe"] != "milieu"]

# 1. Prix médian des proches et des loin, dans chaque commune et chaque année
par_commune = (
    d.groupby(["zone", "nom_commune", "annee", "groupe"])["prix_m2"]
     .agg(mediane="median", n="size")
     .unstack("groupe")
)
par_commune.columns = [f"{stat}_{groupe}" for stat, groupe in par_commune.columns]
par_commune = par_commune.dropna()
par_commune = par_commune[
    (par_commune["n_proche"] >= N_MIN) & (par_commune["n_loin"] >= N_MIN)
].reset_index()

# 2. Écart proche / loin dans chaque commune (%)
par_commune["prime_pct"] = 100 * (par_commune["mediane_proche"] / par_commune["mediane_loin"] - 1)

# 3. Moyenne des écarts sur les communes, pondérée par le plus petit des deux effectifs
par_commune["poids"] = np.minimum(par_commune["n_proche"], par_commune["n_loin"])

def resume(g):
    return pd.Series({
        "prime_commune_egale_pct": np.average(g["prime_pct"], weights=g["poids"]),
        "nb_communes": len(g),
        "part_communes_prime_positive": 100 * (g["prime_pct"] > 0).mean(),
    })

prime = par_commune.groupby(["zone", "annee"]).apply(resume, include_groups=False)

# Pour comparer : l'écart « brut », sans tenir compte de la commune
brut = d.groupby(["zone", "annee", "groupe"])["prix_m2"].median().unstack("groupe")
prime["prime_brute_pct"] = 100 * (brut["proche"] / brut["loin"] - 1)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(prime.round(1))

# Communes aux extrêmes (prime moyenne sur les 5 ans)
classement = par_commune.groupby("nom_commune")["prime_pct"].mean().sort_values().round(1)
print("Primes les plus négatives :\n", classement.head(5))
print("Primes les plus fortes :\n", classement.tail(5))

# Graphique : prime brute vs prime à commune égale
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, zone in zip(axes, ["Paris", "Petite couronne"]):
    p = prime.loc[zone]
    ax.plot(p.index, p["prime_brute_pct"], color="#eb6834", lw=2, marker="o", label="Brute")
    ax.plot(p.index, p["prime_commune_egale_pct"], color="#2a78d6", lw=2, marker="o",
            label="À commune égale")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks([2021, 2022, 2023, 2024, 2025])
    ax.set_title(zone, loc="left")
    ax.set_ylabel(f"Écart de prix < {PROCHE} m vs ≥ {LOIN} m (%)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
fig.suptitle("Prime de proximité : brute et à commune égale", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(dossier / "03_prime_proximite.png", dpi=150)

# ============================================================
# PROJET ENSAE : LE VRAI PRIX DE LA VILLE
#
# RECHERCHE DES INTERVALLES DE DISTANCE LES PLUS VALORISES
#
# Analyse exclusivement descriptive :
# - Prix médians réels
# - Effectifs
# - Comparaison Paris / petite couronne
# - Comparaison intra-communale
# - Evolution annuelle
#
# AUCUNE MODELISATION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# 1. PARAMETRES
# ============================================================

DOSSIER_SCRIPT = Path(__file__).resolve().parent

# Adapter si le CSV est situé ailleurs
FICHIER = Path(
    "/home/onyxia/work/Projet_Data_2A_S1/data/processed/dvf_appartements_stations.csv")

DOSSIER_SORTIE = DOSSIER_SCRIPT / "resultats_distance_optimale"

# Seuil minimal de transactions dans une tranche
MIN_TRANSACTIONS_GLOBAL = 200

# Seuil minimal de transactions par commune et par tranche
MIN_TRANSACTIONS_COMMUNE = 30

# Analyse fine : tranches de 50 mètres jusqu'à 800 mètres
BINS_FINS = list(range(0, 850, 50))
LABELS_FINS = [
    f"{i}-{i+50} m" for i in range(0, 800, 50)
]

# Analyse générale : tranches de 100 mètres jusqu'à 2000 mètres
BINS_LARGES = list(range(0, 2100, 100)) + [np.inf]
LABELS_LARGES = [
    f"{i}-{i+100} m" for i in range(0, 2000, 100)
] + ["2000 m et +"]

REFERENCE = "200-300 m"

sns.set_theme(style="whitegrid")
DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

if not FICHIER.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {FICHIER}\n"
        "Modifie la variable FICHIER avec le chemin exact."
    )


# ============================================================
# 2. CHARGEMENT ET PREPARATION
# ============================================================

print("\nChargement des données DVF...")

df = pd.read_csv(FICHIER, low_memory=False)

colonnes = [
    "prix_m2",
    "dist_station_m",
    "code_commune",
    "nom_commune",
    "code_departement",
    "annee"
]

for col in colonnes:
    if col not in df.columns:
        raise ValueError(f"Colonne manquante : {col}")

df["prix_m2"] = pd.to_numeric(df["prix_m2"], errors="coerce")
df["dist_station_m"] = pd.to_numeric(
    df["dist_station_m"], errors="coerce"
)
df["annee"] = pd.to_numeric(df["annee"], errors="coerce")

df = df.dropna(subset=colonnes).copy()

df = df[
    (df["prix_m2"] > 0) &
    (df["dist_station_m"] >= 0)
].copy()

df["code_departement"] = (
    df["code_departement"].astype(str).str.zfill(2)
)

df["code_commune"] = df["code_commune"].astype(str)

df["zone"] = np.where(
    df["code_departement"] == "75",
    "Paris",
    "Petite couronne"
)

print("Nombre de transactions :", len(df))
print("\nRépartition géographique :")
print(df["zone"].value_counts())


# ============================================================
# 3. ANALYSE GENERALE PAR TRANCHES DE 100 METRES
# ============================================================

print("\nAnalyse par tranches de 100 mètres...")

df["tranche_100m"] = pd.cut(
    df["dist_station_m"],
    bins=BINS_LARGES,
    labels=LABELS_LARGES,
    right=False
)

analyse_100m = (
    df.groupby(["zone", "tranche_100m"], observed=False)
    .agg(
        nombre_transactions=("prix_m2", "size"),
        prix_median=("prix_m2", "median"),
        prix_moyen=("prix_m2", "mean"),
        ecart_type=("prix_m2", "std")
    )
    .reset_index()
)

analyse_100m["effectif_suffisant"] = (
    analyse_100m["nombre_transactions"]
    >= MIN_TRANSACTIONS_GLOBAL
)

analyse_100m.to_csv(
    DOSSIER_SORTIE / "analyse_globale_100m.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 4. IDENTIFICATION DES TRANCHES LES PLUS VALORISEES
# ============================================================

print("\nRecherche des tranches les plus valorisées...")

for zone in ["Paris", "Petite couronne"]:

    temp = analyse_100m[
        (analyse_100m["zone"] == zone) &
        (analyse_100m["effectif_suffisant"])
    ].copy()

    if temp.empty:
        print(f"Aucun intervalle suffisamment représenté : {zone}")
        continue

    meilleur = temp.loc[temp["prix_median"].idxmax()]

    print(f"\n{zone.upper()}")
    print("Intervalle le plus valorisé :", meilleur["tranche_100m"])
    print("Prix médian :", round(meilleur["prix_median"], 2), "€/m²")
    print("Nombre de ventes :", int(meilleur["nombre_transactions"]))


# ============================================================
# 5. ANALYSE FINE PAR TRANCHES DE 50 METRES
# ============================================================

print("\nAnalyse fine par tranches de 50 mètres...")

df["tranche_50m"] = pd.cut(
    df["dist_station_m"],
    bins=BINS_FINS,
    labels=LABELS_FINS,
    right=False
)

analyse_50m = (
    df.groupby(["zone", "tranche_50m"], observed=False)
    .agg(
        nombre_transactions=("prix_m2", "size"),
        prix_median=("prix_m2", "median"),
        prix_moyen=("prix_m2", "mean")
    )
    .reset_index()
)

analyse_50m["effectif_suffisant"] = (
    analyse_50m["nombre_transactions"]
    >= MIN_TRANSACTIONS_GLOBAL
)

analyse_50m.to_csv(
    DOSSIER_SORTIE / "analyse_fine_50m.csv",
    index=False,
    encoding="utf-8-sig"
)

for zone in ["Paris", "Petite couronne"]:

    temp = analyse_50m[
        (analyse_50m["zone"] == zone) &
        (analyse_50m["effectif_suffisant"])
    ]

    if not temp.empty:

        meilleur = temp.loc[temp["prix_median"].idxmax()]

        print(f"\nMeilleure tranche fine - {zone} :")
        print(meilleur["tranche_50m"])
        print("Prix médian :", round(meilleur["prix_median"], 2))
        print("Effectif :", int(meilleur["nombre_transactions"]))


# ============================================================
# 6. ANALYSE INTRA-COMMUNALE PAR TRANCHES DE 50 METRES
# ============================================================

print("\nAnalyse intra-communale fine...")

# On calcule le prix médian de chaque commune et chaque
# tranche de 50 mètres.

prix_commune_50m = (
    df.dropna(subset=["tranche_50m"])
    .groupby(
        ["zone", "code_commune", "nom_commune", "tranche_50m"],
        observed=False
    )
    .agg(
        nombre_transactions=("prix_m2", "size"),
        prix_median=("prix_m2", "median")
    )
    .reset_index()
)

# Conservation des couples commune-tranche suffisamment représentés
prix_commune_50m = prix_commune_50m[
    prix_commune_50m["nombre_transactions"]
    >= MIN_TRANSACTIONS_COMMUNE
].copy()

# Tableau large des prix médians communaux
tableau_prix = prix_commune_50m.pivot_table(
    index=["zone", "code_commune", "nom_commune"],
    columns="tranche_50m",
    values="prix_median",
    observed=False
).reindex(columns=LABELS_FINS)

# Pour la référence 200-300m, on recalcule directement le
# prix médian des ventes situées dans cet intervalle.
# Il ne faut PAS faire la moyenne des médianes 200-250 et 250-300.

ventes_reference = df[
    (df["dist_station_m"] >= 200) &
    (df["dist_station_m"] < 300)
]

reference_commune = (
    ventes_reference
    .groupby(["zone", "code_commune"])
    .agg(
        prix_reference=("prix_m2", "median"),
        n_reference=("prix_m2", "size")
    )
)

reference_commune = reference_commune[
    reference_commune["n_reference"] >= MIN_TRANSACTIONS_COMMUNE
]

resultats_intra = []

for (zone, code_commune, nom_commune), ligne in tableau_prix.iterrows():

    cle = (zone, code_commune)

    if cle not in reference_commune.index:
        continue

    prix_ref = reference_commune.loc[cle, "prix_reference"]

    for tranche in LABELS_FINS:

        prix = ligne[tranche]

        if pd.isna(prix):
            continue

        ecart = ((prix / prix_ref) - 1) * 100

        resultats_intra.append({
            "zone": zone,
            "code_commune": code_commune,
            "nom_commune": nom_commune,
            "tranche_distance": tranche,
            "prix_median": prix,
            "prix_reference": prix_ref,
            "ecart_pourcentage": ecart
        })

ecarts_intra = pd.DataFrame(resultats_intra)

ecarts_intra.to_csv(
    DOSSIER_SORTIE / "ecarts_intra_50m.csv",
    index=False,
    encoding="utf-8-sig"
)

synthese_intra = (
    ecarts_intra
    .groupby(["zone", "tranche_distance"], observed=False)
    .agg(
        nombre_communes=("code_commune", "nunique"),
        ecart_median_pourcentage=("ecart_pourcentage", "median"),
        ecart_moyen_pourcentage=("ecart_pourcentage", "mean"),
        part_communes_positives=(
            "ecart_pourcentage",
            lambda x: (x > 0).mean() * 100
        )
    )
    .reset_index()
)

synthese_intra.to_csv(
    DOSSIER_SORTIE / "synthese_intra_50m.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 7. RECHERCHE DE LA DISTANCE LA PLUS VALORISEE
#    SELON LA COMPARAISON INTRA-COMMUNALE
# ============================================================

print("\nRecherche du maximum intra-communal...")

for zone in ["Paris", "Petite couronne"]:

    temp = synthese_intra[
        (synthese_intra["zone"] == zone) &
        (synthese_intra["nombre_communes"] >= 5)
    ].copy()

    if temp.empty:
        print(f"Pas assez de communes pour {zone}")
        continue

    meilleur = temp.loc[
        temp["ecart_median_pourcentage"].idxmax()
    ]

    print(f"\n{zone.upper()} - ANALYSE INTRA-COMMUNALE")
    print("Tranche la plus valorisée :", meilleur["tranche_distance"])
    print(
        "Écart médian :",
        round(meilleur["ecart_median_pourcentage"], 2),
        "%"
    )
    print("Nombre de communes :", int(meilleur["nombre_communes"]))


# ============================================================
# 8. EVOLUTION ANNUELLE DES PRIX PAR DISTANCE
# ============================================================

print("\nAnalyse annuelle...")

analyse_annuelle = (
    df.groupby(
        ["zone", "annee", "tranche_100m"],
        observed=False
    )
    .agg(
        nombre_transactions=("prix_m2", "size"),
        prix_median=("prix_m2", "median")
    )
    .reset_index()
)

analyse_annuelle["effectif_suffisant"] = (
    analyse_annuelle["nombre_transactions"]
    >= MIN_TRANSACTIONS_GLOBAL
)

analyse_annuelle.to_csv(
    DOSSIER_SORTIE / "evolution_annuelle_100m.csv",
    index=False,
    encoding="utf-8-sig"
)

# Recherche de la meilleure tranche chaque année
meilleurs_annuels = []

for (zone, annee), groupe in analyse_annuelle.groupby(["zone", "annee"]):

    groupe = groupe[groupe["effectif_suffisant"]]

    if groupe.empty:
        continue

    meilleur = groupe.loc[groupe["prix_median"].idxmax()]

    meilleurs_annuels.append({
        "zone": zone,
        "annee": annee,
        "meilleure_tranche": meilleur["tranche_100m"],
        "prix_median": meilleur["prix_median"],
        "nombre_transactions": meilleur["nombre_transactions"]
    })

meilleurs_annuels = pd.DataFrame(meilleurs_annuels)

meilleurs_annuels.to_csv(
    DOSSIER_SORTIE / "meilleure_distance_par_annee.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nMeilleures tranches par année :")
print(meilleurs_annuels.to_string(index=False))


# ============================================================
# 9. GRAPHIQUES
# ============================================================

print("\nCréation des graphiques...")

# ------------------------------------------------------------
# Graphique A : prix médian par tranche de 100m
# ------------------------------------------------------------

for zone in ["Paris", "Petite couronne"]:

    temp = analyse_100m[
        (analyse_100m["zone"] == zone) &
        (analyse_100m["effectif_suffisant"])
    ].copy()

    if temp.empty:
        continue

    plt.figure(figsize=(14, 6))

    plt.plot(
        temp["tranche_100m"].astype(str),
        temp["prix_median"],
        marker="o"
    )

    plt.title(f"Prix médian selon la distance - {zone}")
    plt.xlabel("Distance à la station")
    plt.ylabel("Prix médian (€/m²)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        DOSSIER_SORTIE / f"prix_100m_{zone.replace(' ', '_')}.png",
        dpi=300
    )
    plt.close()


# ------------------------------------------------------------
# Graphique B : analyse fine 50m
# ------------------------------------------------------------

for zone in ["Paris", "Petite couronne"]:

    temp = analyse_50m[
        (analyse_50m["zone"] == zone) &
        (analyse_50m["effectif_suffisant"])
    ].copy()

    if temp.empty:
        continue

    plt.figure(figsize=(14, 6))

    plt.plot(
        temp["tranche_50m"].astype(str),
        temp["prix_median"],
        marker="o"
    )

    plt.title(f"Analyse fine : prix médian tous les 50 mètres - {zone}")
    plt.xlabel("Distance à la station")
    plt.ylabel("Prix médian (€/m²)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        DOSSIER_SORTIE / f"prix_50m_{zone.replace(' ', '_')}.png",
        dpi=300
    )
    plt.close()


# ------------------------------------------------------------
# Graphique C : écart intra-communal
# ------------------------------------------------------------

for zone in ["Paris", "Petite couronne"]:

    temp = synthese_intra[
        synthese_intra["zone"] == zone
    ].copy()

    if temp.empty:
        continue

    plt.figure(figsize=(14, 6))

    plt.bar(
        temp["tranche_distance"].astype(str),
        temp["ecart_median_pourcentage"]
    )

    plt.axhline(0, color="black", linewidth=1)

    plt.title(
        f"Écart médian intra-communal selon la distance - {zone}\n"
        "Référence : 200-300 mètres"
    )

    plt.xlabel("Distance à la station")
    plt.ylabel("Écart médian (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        DOSSIER_SORTIE / f"intra_50m_{zone.replace(' ', '_')}.png",
        dpi=300
    )
    plt.close()


# ------------------------------------------------------------
# Graphique D : évolution annuelle des prix
# ------------------------------------------------------------

for zone in ["Paris", "Petite couronne"]:

    temp = analyse_annuelle[
        (analyse_annuelle["zone"] == zone) &
        (analyse_annuelle["effectif_suffisant"])
    ].copy()

    if temp.empty:
        continue

    plt.figure(figsize=(14, 7))

    sns.lineplot(
        data=temp,
        x="tranche_100m",
        y="prix_median",
        hue="annee",
        marker="o"
    )

    plt.title(f"Evolution annuelle des prix par distance - {zone}")
    plt.xlabel("Distance à la station")
    plt.ylabel("Prix médian (€/m²)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        DOSSIER_SORTIE / f"evolution_annuelle_{zone.replace(' ', '_')}.png",
        dpi=300
    )
    plt.close()


# ============================================================
# 10. RESUME FINAL
# ============================================================

with open(
    DOSSIER_SORTIE / "resume_distance_optimale.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("PROJET ENSAE - LE VRAI PRIX DE LA VILLE\n")
    f.write("RECHERCHE DE LA DISTANCE LA PLUS VALORISEE\n")
    f.write("=" * 65 + "\n\n")

    f.write(f"Transactions analysées : {len(df)}\n")
    f.write(f"Seuil global : {MIN_TRANSACTIONS_GLOBAL}\n")
    f.write(f"Seuil par commune : {MIN_TRANSACTIONS_COMMUNE}\n\n")

    f.write("MEILLEURES TRANCHES DE 100 METRES\n")
    f.write("-" * 65 + "\n")

    for zone in ["Paris", "Petite couronne"]:

        temp = analyse_100m[
            (analyse_100m["zone"] == zone) &
            (analyse_100m["effectif_suffisant"])
        ]

        if not temp.empty:

            meilleur = temp.loc[temp["prix_median"].idxmax()]

            f.write(
                f"{zone} : {meilleur['tranche_100m']} - "
                f"{meilleur['prix_median']:.2f} €/m² "
                f"({int(meilleur['nombre_transactions'])} ventes)\n"
            )

    f.write("\nMEILLEURES TRANCHES EN ANALYSE INTRA-COMMUNALE\n")
    f.write("-" * 65 + "\n")

    for zone in ["Paris", "Petite couronne"]:

        temp = synthese_intra[
            (synthese_intra["zone"] == zone) &
            (synthese_intra["nombre_communes"] >= 5)
        ]

        if not temp.empty:

            meilleur = temp.loc[
                temp["ecart_median_pourcentage"].idxmax()
            ]

            f.write(
                f"{zone} : {meilleur['tranche_distance']} - "
                f"{meilleur['ecart_median_pourcentage']:.2f}% "
                f"({int(meilleur['nombre_communes'])} communes)\n"
            )

    f.write("\nLIMITES METHODOLOGIQUES\n")
    f.write("-" * 65 + "\n")

    f.write(
        "Le prix médian le plus élevé identifie une tranche "
        "de distance la plus valorisée dans les transactions observées. "
        "Il ne démontre pas l'existence d'une distance optimale causale.\n"
    )

    f.write(
        "Les effectifs doivent être examinés pour chaque intervalle. "
        "Les communes contributrices peuvent varier selon les distances.\n"
    )

    f.write(
        "La comparaison intra-communale réduit les différences "
        "entre communes, mais ne neutralise pas les différences "
        "de quartier ou les caractéristiques des logements.\n"
    )

print("\n" + "=" * 65)
print("ANALYSE TERMINEE")
print("=" * 65)

print("\nRésultats enregistrés dans :", DOSSIER_SORTIE)
print("\nAucune modélisation n'a été effectuée.")

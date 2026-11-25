
# ============================================================
# PROJET ENSAE : LE VRAI PRIX DE LA VILLE
# Analyse descriptive intra-communale
#
# Objectif :
# Comparer les prix au m² selon la distance aux stations,
# à commune identique.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. PARAMETRES
# ------------------------------------------------------------

DOSSIER_SORTIE = "resultats_analyse_intra_commune"
FICHIER = "/home/onyxia/work/Projet_Data_2A_S1/data/processed/dvf_appartements_stations.csv"

# Nombre minimal de transactions par commune et par tranche
MIN_TRANSACTIONS = 30

# Nombre minimal de tranches représentées dans une commune
MIN_TRANCHES = 3

# Tranches de distance en mètres
BINS = [0, 100, 200, 300, 400, 600, 800, 1000, 1500, np.inf]

LABELS = [
    "0-100 m",
    "100-200 m",
    "200-300 m",
    "300-400 m",
    "400-600 m",
    "600-800 m",
    "800-1000 m",
    "1000-1500 m",
    "1500 m et +"
]

# Création du dossier de résultats
os.makedirs(DOSSIER_SORTIE, exist_ok=True)

sns.set_theme(style="whitegrid")


# ------------------------------------------------------------
# 2. CHARGEMENT ET VERIFICATION DES DONNEES
# ------------------------------------------------------------

print("\nChargement des données...")

df = pd.read_csv(FICHIER, low_memory=False)

print("Dimensions initiales :", df.shape)
print("Colonnes :", list(df.columns))

colonnes_requises = [
    "prix_m2",
    "dist_station_m",
    "code_commune",
    "nom_commune",
    "code_departement",
    "annee"
]

for col in colonnes_requises:
    if col not in df.columns:
        raise ValueError(f"Colonne manquante : {col}")

# Conversion numérique
for col in ["prix_m2", "dist_station_m", "annee"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Suppression des observations inexploitables
df = df.dropna(subset=colonnes_requises).copy()

# Prix et distances positifs
df = df[
    (df["prix_m2"] > 0) &
    (df["dist_station_m"] >= 0)
].copy()

# Identification de la zone
df["zone"] = np.where(
    df["code_departement"].astype(str).str.zfill(2) == "75",
    "Paris",
    "Petite couronne"
)

# Identifiant communal robuste
df["code_commune"] = df["code_commune"].astype(str)

# Construction des tranches de distance
df["tranche_distance"] = pd.cut(
    df["dist_station_m"],
    bins=BINS,
    labels=LABELS,
    right=False
)

df = df.dropna(subset=["tranche_distance"]).copy()

print("\nNombre de transactions exploitables :", len(df))
print("\nRépartition par zone :")
print(df["zone"].value_counts())


# ------------------------------------------------------------
# 3. STATISTIQUES DESCRIPTIVES GENERALES
# ------------------------------------------------------------

print("\nCalcul des statistiques générales...")

statistiques_generales = (
    df.groupby(["zone", "tranche_distance"], observed=False)
    .agg(
        nombre_transactions=("prix_m2", "size"),
        prix_median=("prix_m2", "median"),
        prix_moyen=("prix_m2", "mean"),
        ecart_type=("prix_m2", "std")
    )
    .reset_index()
)

statistiques_generales.to_csv(
    os.path.join(DOSSIER_SORTIE, "statistiques_generales.csv"),
    index=False,
    encoding="utf-8-sig"
)

print(statistiques_generales.to_string(index=False))


# ------------------------------------------------------------
# 4. PRIX MEDIAN PAR COMMUNE ET PAR DISTANCE
# ------------------------------------------------------------

print("\nCalcul des prix médians par commune...")

# Chaque ligne correspond à une commune et une tranche
commune_distance = (
    df.groupby(
        ["zone", "code_commune", "nom_commune", "tranche_distance"],
        observed=False
    )
    .agg(
        nombre_transactions=("prix_m2", "size"),
        prix_median=("prix_m2", "median"),
        prix_moyen=("prix_m2", "mean")
    )
    .reset_index()
)

commune_distance.to_csv(
    os.path.join(DOSSIER_SORTIE, "prix_commune_distance.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 5. SELECTION DES COMMUNES SUFFISAMMENT REPRESENTEES
# ------------------------------------------------------------

print("\nSélection des communes comparables...")

# On conserve les couples commune-tranche ayant au moins
# MIN_TRANSACTIONS transactions.

commune_distance_filtre = commune_distance[
    commune_distance["nombre_transactions"] >= MIN_TRANSACTIONS
].copy()

# Nombre de tranches suffisamment représentées par commune
couverture = (
    commune_distance_filtre
    .groupby(["zone", "code_commune"])
    ["tranche_distance"]
    .nunique()
    .reset_index(name="nombre_tranches")
)

# Communes ayant au moins MIN_TRANCHES tranches représentées
communes_retenues = couverture[
    couverture["nombre_tranches"] >= MIN_TRANCHES
].copy()

commune_distance_filtre = commune_distance_filtre.merge(
    communes_retenues[["zone", "code_commune"]],
    on=["zone", "code_commune"],
    how="inner"
)

print("\nNombre de communes retenues par zone :")
print(
    communes_retenues.groupby("zone")["code_commune"]
    .nunique()
)

commune_distance_filtre.to_csv(
    os.path.join(DOSSIER_SORTIE, "communes_retenues.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 6. CONSTRUCTION DU TABLEAU COMPARATIF
# ------------------------------------------------------------

print("\nConstruction du tableau comparatif...")

# Tableau large :
# une ligne = une commune
# une colonne = une tranche de distance

tableau_prix = commune_distance_filtre.pivot_table(
    index=["zone", "code_commune"],
    columns="tranche_distance",
    values="prix_median",
    observed=False
)

# On conserve les communes ayant au moins MIN_TRANCHES
# médianes disponibles après filtrage.
tableau_prix = tableau_prix.dropna(
    thresh=MIN_TRANCHES
)

tableau_prix.to_csv(
    os.path.join(DOSSIER_SORTIE, "tableau_prix_par_commune.csv"),
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 7. COMPARAISON DES ECARTS AU SEIN DES COMMUNES
# ------------------------------------------------------------

print("\nCalcul des écarts de prix intra-communaux...")

# Référence : tranche 200-300 mètres.
# Ce choix permet de comparer les autres tranches à une
# distance intermédiaire, plutôt que de prendre la tranche
# la plus proche comme référence.

REFERENCE = "200-300 m"

if REFERENCE not in tableau_prix.columns:
    raise ValueError("La tranche de référence est absente.")

# On garde les communes ayant une médiane de référence
# disponible.
tableau_comparable = tableau_prix.dropna(
    subset=[REFERENCE]
).copy()

# Calcul des écarts en euros et en pourcentage
resultats_ecarts = []

for (zone, code_commune), ligne in tableau_comparable.iterrows():

    prix_reference = ligne[REFERENCE]

    for tranche in LABELS:

        if tranche not in ligne.index:
            continue

        prix_tranche = ligne[tranche]

        if pd.isna(prix_tranche):
            continue

        ecart_euros = prix_tranche - prix_reference

        ecart_pourcentage = (
            (prix_tranche / prix_reference) - 1
        ) * 100

        resultats_ecarts.append({
            "zone": zone,
            "code_commune": code_commune,
            "tranche_distance": tranche,
            "prix_median_tranche": prix_tranche,
            "prix_median_reference": prix_reference,
            "ecart_euros": ecart_euros,
            "ecart_pourcentage": ecart_pourcentage
        })

ecarts = pd.DataFrame(resultats_ecarts)

ecarts.to_csv(
    os.path.join(DOSSIER_SORTIE, "ecarts_intra_commune.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 8. AGREGER LES ECARTS ENTRE COMMUNES
# ------------------------------------------------------------

print("\nAgrégation des écarts entre communes...")

# Chaque commune a ici le même poids dans la médiane.
# On évite ainsi que les communes ayant beaucoup de ventes
# dominent mécaniquement le résultat.

synthese_intra = (
    ecarts.groupby(["zone", "tranche_distance"], observed=False)
    .agg(
        nombre_communes=("code_commune", "nunique"),
        ecart_median_pourcentage=("ecart_pourcentage", "median"),
        ecart_moyen_pourcentage=("ecart_pourcentage", "mean"),
        ecart_median_euros=("ecart_euros", "median"),
        part_communes_positives=(
            "ecart_pourcentage",
            lambda x: (x > 0).mean() * 100
        )
    )
    .reset_index()
)

synthese_intra.to_csv(
    os.path.join(DOSSIER_SORTIE, "synthese_intra_commune.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n========== SYNTHESE INTRA-COMMUNALE ==========\n")
print(synthese_intra.to_string(index=False))


# ------------------------------------------------------------
# 9. GRAPHIQUE 1 : ECART MEDIAN DE PRIX PAR DISTANCE
# ------------------------------------------------------------

print("\nCréation du graphique 1...")

for zone in ["Paris", "Petite couronne"]:

    donnees = synthese_intra[
        synthese_intra["zone"] == zone
    ].copy()

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=donnees,
        x="tranche_distance",
        y="ecart_median_pourcentage"
    )

    plt.axhline(
        0,
        color="black",
        linewidth=1
    )

    plt.title(
        f"Écart médian de prix au m² intra-communal - {zone}\n"
        f"Référence : {REFERENCE}"
    )

    plt.xlabel("Distance à la station la plus proche")
    plt.ylabel("Écart médian par rapport à la référence (%)")

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            DOSSIER_SORTIE,
            f"graphique_ecarts_intra_{zone.replace(' ', '_')}.png"
        ),
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# 10. GRAPHIQUE 2 : PART DES COMMUNES AVEC UN ECART POSITIF
# ------------------------------------------------------------

print("\nCréation du graphique 2...")

for zone in ["Paris", "Petite couronne"]:

    donnees = synthese_intra[
        synthese_intra["zone"] == zone
    ].copy()

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=donnees,
        x="tranche_distance",
        y="part_communes_positives"
    )

    plt.axhline(
        50,
        color="red",
        linestyle="--",
        label="50 % des communes"
    )

    plt.title(
        f"Part des communes où le prix dépasse la référence - {zone}"
    )

    plt.xlabel("Distance à la station la plus proche")
    plt.ylabel("Communes avec un écart positif (%)")

    plt.xticks(rotation=35, ha="right")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            DOSSIER_SORTIE,
            f"graphique_part_communes_{zone.replace(' ', '_')}.png"
        ),
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# 11. GRAPHIQUE 3 : DISTRIBUTION DES ECARTS PAR COMMUNE
# ------------------------------------------------------------

print("\nCréation du graphique 3...")

for zone in ["Paris", "Petite couronne"]:

    donnees = ecarts[
        (ecarts["zone"] == zone) &
        (ecarts["tranche_distance"] != REFERENCE)
    ].copy()

    plt.figure(figsize=(13, 7))

    sns.boxplot(
        data=donnees,
        x="tranche_distance",
        y="ecart_pourcentage",
        showfliers=False
    )

    plt.axhline(
        0,
        color="red",
        linestyle="--"
    )

    plt.title(
        f"Distribution des écarts de prix entre communes - {zone}\n"
        f"Référence : {REFERENCE}"
    )

    plt.xlabel("Distance à la station")
    plt.ylabel("Écart de prix par rapport à la référence (%)")

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            DOSSIER_SORTIE,
            f"graphique_distribution_{zone.replace(' ', '_')}.png"
        ),
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# 12. COMPARAISON PONDEREE PAR LE NOMBRE DE TRANSACTIONS
# ------------------------------------------------------------

print("\nCalcul de la synthèse pondérée par les transactions...")

# Cette seconde synthèse donne davantage de poids aux communes
# ayant davantage de transactions dans chaque tranche.
# Elle est complémentaire de la médiane à poids égal par commune.

ecarts_avec_effectifs = ecarts.merge(
    commune_distance_filtre[
        ["zone", "code_commune", "tranche_distance",
         "nombre_transactions"]
    ],
    on=["zone", "code_commune", "tranche_distance"],
    how="left"
)

synthese_ponderee = (
    ecarts_avec_effectifs
    .groupby(["zone", "tranche_distance"], observed=False)
    .apply(
        lambda groupe: pd.Series({
            "nombre_communes": groupe["code_commune"].nunique(),
            "ecart_moyen_pondere": np.average(
                groupe["ecart_pourcentage"],
                weights=groupe["nombre_transactions"]
            ) if groupe["nombre_transactions"].notna().any() else np.nan
        }),
        include_groups=False
    )
    .reset_index()
)

synthese_ponderee.to_csv(
    os.path.join(DOSSIER_SORTIE, "synthese_ponderee.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 13. RESUME AUTOMATIQUE
# ------------------------------------------------------------

print("\nGénération du résumé...")

fichier_resume = os.path.join(
    DOSSIER_SORTIE,
    "resume_analyse.txt"
)

with open(fichier_resume, "w", encoding="utf-8") as f:

    f.write("PROJET ENSAE - LE VRAI PRIX DE LA VILLE\n")
    f.write("ANALYSE DESCRIPTIVE INTRA-COMMUNALE\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Nombre de transactions analysées : {len(df)}\n")
    f.write(
        f"Nombre de communes retenues : "
        f"{communes_retenues['code_commune'].nunique()}\n\n"
    )

    f.write(f"Tranche de référence : {REFERENCE}\n")
    f.write(f"Minimum de transactions par tranche : {MIN_TRANSACTIONS}\n")
    f.write(f"Minimum de tranches par commune : {MIN_TRANCHES}\n\n")

    f.write("SYNTHESE DES ECARTS INTRA-COMMUNAUX\n")
    f.write("-" * 60 + "\n")

    f.write(synthese_intra.to_string(index=False))

    f.write("\n\nINTERPRETATION METHODOLOGIQUE\n")
    f.write(
        "Cette analyse compare les prix médians des appartements "
        "selon leur distance à une station, au sein d'une même commune.\n"
    )
    f.write(
        "Les écarts sont ensuite agrégés entre communes avec un poids "
        "égal pour chaque commune dans la médiane.\n"
    )
    f.write(
        "Il s'agit d'une analyse descriptive et non causale. "
        "La commune ne contrôle pas toutes les différences de quartier, "
        "de rue, de qualité du logement ou d'environnement.\n"
    )


# ------------------------------------------------------------
# 14. FIN
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("ANALYSE TERMINEE AVEC SUCCES")
print("=" * 60)

print(f"\nTous les résultats sont enregistrés dans : {DOSSIER_SORTIE}")

print("\nFichiers principaux générés :")
print("- statistiques_generales.csv")
print("- prix_commune_distance.csv")
print("- communes_retenues.csv")
print("- tableau_prix_par_commune.csv")
print("- ecarts_intra_commune.csv")
print("- synthese_intra_commune.csv")
print("- synthese_ponderee.csv")
print("- resume_analyse.txt")

print("\nGraphiques générés :")
print("- écarts intra-communaux")
print("- part des communes avec un écart positif")
print("- distribution des écarts par commune")

print("\nAucune modélisation statistique n'a été effectuée.")

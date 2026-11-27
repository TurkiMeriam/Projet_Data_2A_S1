"""Graphiques du projet. Chaque fonction enregistre la figure dans outputs/figures."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .analysis import REFERENCE, ZONES

FIG = config.ROOT / "outputs" / "figures"


def style() -> None:
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3})


def _sauver(fig, nom: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / nom, bbox_inches="tight")


def prix_zone_annee(prix: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    prix.plot(ax=ax, marker="o")
    ax.set_xticks(prix.index)
    ax.set(xlabel="Année", ylabel="Prix médian (€/m²)", title="Prix médian au m² par zone")
    _sauver(fig, "prix_par_zone_annee.png")
    return fig


def carte_prix(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(df["longitude"], df["latitude"], C=df["prix_m2"], reduce_C_function=np.median,
                   gridsize=90, mincnt=10, cmap="viridis")
    ax.set_aspect(1 / np.cos(np.radians(48.85)))  # corrige la déformation longitude / latitude
    ax.grid(False)
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Prix médian au m² par cellule (≥ 10 ventes)")
    fig.colorbar(hb, label="€/m²")
    _sauver(fig, "carte_prix.png")
    return fig


def prix_selon_distance(fin: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for zone, g in fin.groupby("zone"):
        ax.plot(g["distance_m"], g["mediane"], marker="o", ms=3, label=zone)
    ax.set(xlabel="Distance à la station la plus proche (m)", ylabel="Prix médian (€/m²)",
           title="Prix médian selon la distance à la station")
    ax.legend(frameon=False)
    _sauver(fig, "prix_selon_distance.png")
    return fig


# --------------------------------------------------------------------------- exploration descriptive

COULEURS = {"Paris": "#1f77b4", "Petite couronne": "#e67e22"}


def distribution_prix(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for zone, g in df.groupby("zone"):
        axes[0].hist(g["prix_m2"], bins=60, density=True, histtype="step", label=zone, color=COULEURS[zone])
    axes[0].set(title="Distribution des prix au m²", xlabel="Prix au m² (€)")
    axes[0].legend(frameon=False)
    axes[1].boxplot([g["prix_m2"] for _, g in df.groupby("zone")], tick_labels=sorted(df["zone"].unique()))
    axes[1].set(title="Prix par zone", ylabel="Prix au m² (€)")
    _sauver(fig, "distribution_prix.png")
    return fig


def courbe_tranches(table: pd.DataFrame, nom: str, titre: str, max_tranches: int | None = None):
    """Prix médian par tranche, une courbe par zone."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for zone, g in table.groupby("zone"):
        g = g.head(max_tranches) if max_tranches else g
        ax.plot(g["tranche"].astype(str), g["prix_median"], marker="o", label=zone, color=COULEURS[zone])
    ax.set(xlabel="Distance à la station (m)", ylabel="Prix médian (€/m²)", title=titre)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False)
    _sauver(fig, nom)
    return fig


def courbes_par_annee(table: pd.DataFrame, colonne: str, nom: str, ylabel: str):
    """Une figure par zone, une courbe par année."""
    figs = []
    for zone, g in table.groupby("zone"):
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for annee, h in g.groupby("annee"):
            ax.plot(h["tranche"].astype(str), h[colonne], marker="o", label=str(annee))
        if "evolution" in colonne:
            ax.axhline(0, color="black", lw=0.8)
        ax.set(xlabel="Distance à la station (m)", ylabel=ylabel, title=zone)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Année", frameon=False, ncols=5)
        _sauver(fig, f"{nom}_{zone.lower().replace(' ', '_')}.png")
        figs.append(fig)
    return figs


def ecarts_communes_barres(synthese: pd.DataFrame):
    """Écart médian à la tranche de référence, par tranche (une figure par zone)."""
    figs = []
    for zone, g in synthese.groupby("zone"):
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.bar(g["tranche"].astype(str), g["ecart_median_pct"], color=COULEURS[zone])
        ax.axhline(0, color="black", lw=0.8)
        ax.set(xlabel="Distance à la station (m)", ylabel="Écart médian vs 200-400 m (%)", title=zone)
        ax.tick_params(axis="x", rotation=55)
        _sauver(fig, f"ecarts_communes_{zone.lower().replace(' ', '_')}.png")
        figs.append(fig)
    return figs


def sensibilite_seuils(rob: pd.DataFrame):
    figs = []
    for zone, g in rob.groupby("zone"):
        fig, ax = plt.subplots(figsize=(12, 4.5))
        for seuil, h in g.groupby("seuil_effectif"):
            ax.plot(h["tranche"].astype(str), h["ecart_median_pct"], marker="o", label=f"seuil {seuil}")
        ax.axhline(0, color="black", lw=0.8)
        ax.set(xlabel="Distance à la station (m)", ylabel="Écart médian (%)", title=zone)
        ax.tick_params(axis="x", rotation=55)
        ax.legend(frameon=False)
        _sauver(fig, f"sensibilite_seuils_{zone.lower().replace(' ', '_')}.png")
        figs.append(fig)
    return figs


def courbe_globale(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(table["tranche"].astype(str), table["prix_median"], marker="o", lw=2.2)
    ax.set(xlabel="Distance à la station (m)", ylabel="Prix médian (€/m²)",
           title="Prix médian selon la distance à la station (toutes zones)")
    ax.tick_params(axis="x", rotation=35)
    _sauver(fig, "prix_selon_distance_global.png")
    return fig


def courbe_modes(courbe: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for mode, g in courbe.groupby("mode_station"):
        ax.plot(g["tranche"].astype(str), g["mediane"], marker="o", lw=2.2, label=mode)
    ax.set(xlabel="Tranche de distance à la station (m)", ylabel="Prix médian (€/m²)",
           title="Prix médian selon la distance : métro contre RER")
    ax.legend(frameon=False)
    _sauver(fig, "prix_metro_vs_rer.png")
    return fig


# --------------------------------------------------------------------------- effet de la distance sans modèle

def prix_relatif_tranches(var: pd.DataFrame):
    """Prix médian relatif à la commune (écart en %) selon la distance, une courbe par zone."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for zone, g in var.groupby("zone"):
        ax.plot(g["tranche"], g["prix_relatif_median_pct"], marker="o", label=zone, color=COULEURS[zone])
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="Distance à la station (m)", ylabel="Écart au prix médian de la commune (%)",
           title="Prix relatif à la commune selon la distance")
    ax.legend(frameon=False)
    _sauver(fig, "prix_relatif_distance.png")
    return fig


def ecarts_stratifies_barres(res: pd.DataFrame):
    """Écart moyen pondéré à la tranche de référence ; barres d'erreur = IC à 95 % si `res` les contient
    (sortie de `analysis.ic_ecarts`)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    avec_ic = {"ic_bas_moyenne", "ic_haut_moyenne"} <= set(res.columns)
    for ax, zone in zip(axes, ZONES):
        g = res[res["zone"] == zone]
        err = None
        if avec_ic:
            err = np.vstack([g["ecart_pct"] - g["ic_bas_moyenne"], g["ic_haut_moyenne"] - g["ecart_pct"]])
        ax.bar(g["tranche"].astype(str), g["ecart_pct"], color=COULEURS[zone], yerr=err, capsize=4,
               error_kw={"lw": 1, "ecolor": "black"})
        ax.axhline(0, color="black", lw=0.8)
        ax.set(title=zone, xlabel="Distance à la station (m)")
    axes[0].set_ylabel(f"Écart de prix vs {REFERENCE} m (%)")
    if avec_ic:
        fig.suptitle("Écart à biens comparables (barres : intervalle de confiance à 95 %, bootstrap par commune)",
                     fontsize=10)
    _sauver(fig, "ecarts_stratifies.png")
    return fig


def courbe_modes_par_zone(table: pd.DataFrame):
    """Prix médian par tranche de distance pour le métro et le RER, à zone constante (un panneau par zone)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, zone in zip(axes, ZONES):
        g = table[table["zone"] == zone]
        for mode, style in [("METRO", "-o"), ("RER", "--s")]:
            col = f"mediane_{mode}"
            if col in g:
                h = g.dropna(subset=[col])
                ax.plot(h["tranche"].astype(str), h[col], style, color=COULEURS[zone], label=mode)
        ax.set(title=zone, xlabel="Distance à la station (m)")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Prix médian (€/m²)")
    _sauver(fig, "prix_metro_vs_rer_par_zone.png")
    return fig


def prime_stratifiee_annee(res: pd.DataFrame):
    """Prime proche/loin par année. Zone grisée : IC à 95 % (bootstrap par commune) si `res` vient de
    `analysis.ic_prime(..., par_annee=True)`, sinon moitié centrale des cellules (q25-q75)."""
    avec_ic = {"ic_bas_moyenne", "ic_haut_moyenne"} <= set(res.columns)
    bas, haut = ("ic_bas_moyenne", "ic_haut_moyenne") if avec_ic else ("q25_%", "q75_%")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for zone, g in res.groupby("zone"):
        annees = g["annee"].astype(int)
        ax.plot(annees, g["prime_%"], marker="o", label=zone, color=COULEURS[zone])
        ax.fill_between(annees, g[bas], g[haut], alpha=0.15, color=COULEURS[zone])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(sorted(res["annee"].astype(int).unique()))
    legende = "IC à 95 %" if avec_ic else "50 % des cellules"
    ax.set(xlabel="Année", ylabel="Prime de proximité (%)",
           title=f"Biens < 300 m contre ≥ 600 m (zone grisée : {legende})")
    ax.legend(frameon=False)
    _sauver(fig, "prime_stratifiee_annee.png")
    return fig


def evolution_prolongements(prol: pd.DataFrame):
    """Indice du prix médian (base 100 la première année) : quartiers desservis en 2024 contre témoins."""
    from .analysis import DESSERVI, TEMOIN

    fig, ax = plt.subplots(figsize=(8, 4.5))
    annees = prol.index.to_numpy()
    for grp, couleur in [(DESSERVI, "#e67e22"), (TEMOIN, "#7f8c8d")]:
        y = prol[f"indice {grp}"]
        ax.plot(annees, y, marker="o", ms=6, lw=2, color=couleur, label=grp)
        ax.annotate(f"{y.iloc[-1]:.1f}", (annees[-1], y.iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.axhline(100, color="black", lw=0.8)
    ax.axvline(2024.45, color="grey", ls="--", lw=1)  # juin 2024
    ax.text(2024.5, ax.get_ylim()[1], "ouverture\n(juin 2024)", va="top", fontsize=8, color="grey")
    ax.set_xticks(annees)
    ax.set(xlabel="Année", ylabel=f"Prix médian au m² (base 100 en {annees[0]})",
           title="Prolongements des lignes 11 et 14 : quartiers desservis contre témoins")
    ax.legend(frameon=False, loc="lower left")
    _sauver(fig, "prolongements_l11_l14.png")
    return fig

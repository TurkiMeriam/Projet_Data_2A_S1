import gzip

import pandas as pd

from vrai_prix import dvf


def row(id_mut, type_local="Appartement", nature="Vente", valeur=500_000.0,
        surface=50.0, pieces=2, lat=48.85, lon=2.35, disp=1, date="2023-05-10", lot="1"):
    return dict(
        id_mutation=id_mut, date_mutation=date, numero_disposition=disp,
        nature_mutation=nature, valeur_fonciere=valeur, adresse_numero="10",
        adresse_nom_voie="RUE X", code_postal="75011", code_commune="75111",
        nom_commune="Paris 11e Arrondissement", code_departement="75",
        id_parcelle="75111000AA0001", type_local=type_local,
        surface_reelle_bati=surface, nombre_pieces_principales=pieces,
        lot1_numero=lot, lot2_numero=None, longitude=lon, latitude=lat,
    )


def make_raw() -> pd.DataFrame:
    rows = [
        row("ok1"),                                           # valide : 10 000 €/m²
        row("ok1"),                                           # doublon strict
        row("dep", valeur=300_000, surface=40),               # appart + dépendance
        row("dep", type_local="Dépendance", surface=0, valeur=300_000),
        row("vefa", nature="Vente en l'état futur d'achèvement"),
        row("deux_apparts", lot="1"), row("deux_apparts", lot="2"),             # multi-biens
        row("maison_et_appart"), row("maison_et_appart", type_local="Maison"),
        row("cheap", valeur=20_000),                          # 400 €/m²
        row("luxe", valeur=5_000_000),                        # 100 000 €/m²
        row("minuscule", surface=3, valeur=40_000),           # surface < 9 m²
        row("sans_geo", lat=None, lon=None),
        row("hors_idf", lat=43.3, lon=5.4),                   # Marseille
        row("multi_disp", disp=1), row("multi_disp", disp=2),
        row("pieces0", pieces=0, valeur=400_000, surface=40),
        row("terrain", type_local=None),                      # ligne sans local
    ]
    return pd.DataFrame(rows).assign(date_mutation=lambda d: pd.to_datetime(d["date_mutation"]))


def test_clean_keeps_only_simple_apartment_sales():
    out, rep = dvf.clean_dvf(make_raw())
    assert set(out["id_mutation"]) == {"ok1", "dep", "pieces0"}
    assert out["id_mutation"].is_unique
    assert rep["mutations_finales"] == 3


def test_prix_m2_and_derived_columns():
    out, _ = dvf.clean_dvf(make_raw())
    ok1 = out.set_index("id_mutation").loc["ok1"]
    assert ok1["prix_m2"] == 10_000
    assert ok1["annee"] == 2023
    assert abs(ok1["log_prix_m2"] - 9.2103) < 1e-3
    assert ok1["adresse"] == "10 RUE X"


def test_dependances_counted_and_pieces_zero_is_missing():
    out, _ = dvf.clean_dvf(make_raw())
    idx = out.set_index("id_mutation")
    assert idx.loc["dep", "n_dependances"] == 1
    assert idx.loc["ok1", "n_dependances"] == 0
    assert pd.isna(idx.loc["pieces0", "nombre_pieces_principales"])


def test_read_dvf_gz_roundtrip(tmp_path):
    raw = make_raw()
    p = tmp_path / "75.csv.gz"
    with gzip.open(p, "wt") as f:
        raw.assign(colonne_inutile=1).to_csv(f, index=False)
    back = dvf.read_dvf(p)
    assert "colonne_inutile" not in back.columns
    out, _ = dvf.clean_dvf(back)
    assert len(out) == 3


def test_build_dataset_skip_download(tmp_path):
    p = dvf.raw_path(2023, "75", tmp_path / "raw")
    p.parent.mkdir(parents=True)
    with gzip.open(p, "wt") as f:
        make_raw().to_csv(f, index=False)
    out_path = tmp_path / "out" / "dvf.parquet"
    res = dvf.build_dataset([2023], ["75"], tmp_path / "raw", out_path, download=False)
    assert len(res) == 3
    assert len(pd.read_parquet(out_path)) == 3
    assert (tmp_path / "out" / "dvf_rapport.csv").exists()


def test_url_template():
    assert dvf.dvf_url(2024, "92").endswith("/2024/departements/92.csv.gz")

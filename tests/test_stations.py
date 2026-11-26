import numpy as np
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from vrai_prix import stations

# Châtelet (Paris) ; constantes approximatives « mètres par degré »
LON0, LAT0 = 2.3470, 48.8584
M_PER_DEG_LON, M_PER_DEG_LAT = 73_000, 111_200


def pt(dx_m=0.0, dy_m=0.0):
    return LON0 + dx_m / M_PER_DEG_LON, LAT0 + dy_m / M_PER_DEG_LAT


def raw_idfm() -> gpd.GeoDataFrame:
    """Imite le GeoJSON IDFM : une ligne par (station, ligne)."""
    rows = [
        # id, nom, ligne, mode, date_mise, position
        (1, "Alpha", "METRO 1", "METRO", "1900-01-01", pt(0, 0)),
        (1, "Alpha", "RER A", "RER", "1977-12-09", pt(10, 0)),        # même station (zdc), 2 lignes
        (2, "Beta", "METRO 14", "METRO", "2024-06-24", pt(1000, 0)),  # ouverte en 2024
        (3, "Gamma", "TRAM T3b", "TRAMWAY", "2012-12-15", pt(0, 500)),  # mode exclu par défaut
        (4, "Delta", "METRO 4", "METRO", None, pt(-2000, 0)),         # date manquante
        (5, "Epsilon", "METRO 9", "METRO", "1950-01-01", None),       # sans géométrie
    ]
    return gpd.GeoDataFrame(
        {
            "id_ref_zdc": [r[0] for r in rows],
            "nom_zdc": [r[1] for r in rows],
            "res_com": [r[2] for r in rows],
            "mode": [r[3] for r in rows],
            "date_mise": pd.to_datetime([r[4] for r in rows], utc=True),
        },
        geometry=[Point(*r[5]) if r[5] else None for r in rows],
        crs="EPSG:4326",
    )


@pytest.fixture
def lignes():
    return stations.prepare_stations_lignes(raw_idfm(), modes=["METRO", "RER"])


# ---------------------------------------------------------------- préparation
def test_prepare_filters_modes_and_geometry(lignes):
    assert set(lignes["mode"]) == {"METRO", "RER"}          # tram exclu
    assert len(lignes) == 4                                  # Epsilon (sans géométrie) exclu
    assert list(lignes.columns) == ["id_station", "nom_station", "ligne", "mode", "date_mise", "geometry"]
    assert lignes.crs.to_epsg() == 2154


def test_prepare_dates_and_ids(lignes):
    assert lignes["date_mise"].dt.tz is None                 # dates sans fuseau
    beta = lignes[lignes["id_station"] == "2"].iloc[0]
    assert beta["date_mise"] == pd.Timestamp("2024-06-24")
    assert lignes[lignes["id_station"] == "4"]["date_mise"].isna().all()


def test_prepare_drops_duplicate_station_line_pairs():
    raw = raw_idfm()
    doublon = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    out = stations.prepare_stations_lignes(doublon, modes=["METRO", "RER"])
    assert len(out) == 4


def test_prepare_raises_when_no_station_left():
    with pytest.raises(ValueError, match="Aucune station"):
        stations.prepare_stations_lignes(raw_idfm(), modes=["CABLE"])


def test_read_stations_lignes_from_geojson(tmp_path):
    p = tmp_path / "s.geojson"
    raw = raw_idfm()
    raw[raw.geometry.notna()].to_file(p, driver="GeoJSON")
    out = stations.read_stations_lignes(p, modes=["METRO", "RER"])
    assert len(out) == 4 and out.crs.to_epsg() == 2154


def test_download_stations_skips_existing_file(tmp_path):
    dest = tmp_path / "x.geojson"
    dest.write_text("{}")
    assert stations.download_stations(dest) == dest          # aucun appel réseau


# ---------------------------------------------------------- réseau à une date
def test_stations_ouvertes_before_2024(lignes):
    out = stations.stations_ouvertes(lignes, pd.Timestamp("2020-01-01"))
    assert set(out["id_station"]) == {"1", "4"}              # Beta pas encore ouverte, Delta (sans date) oui
    alpha = out.set_index("id_station").loc["1"]
    assert alpha["nb_lignes_station"] == 2
    assert alpha["modes_station"] == "METRO+RER"
    assert alpha["lignes_station"] == "METRO 1, RER A"
    assert alpha["date_ouverture_station"] == pd.Timestamp("1900-01-01")
    assert out.crs.to_epsg() == 2154


def test_stations_ouvertes_includes_opening_day(lignes):
    out = stations.stations_ouvertes(lignes, pd.Timestamp("2024-06-24"))
    assert set(out["id_station"]) == {"1", "2", "4"}


def test_station_point_is_mean_of_its_lines(lignes):
    out = stations.stations_ouvertes(lignes, pd.Timestamp("2020-01-01")).set_index("id_station")
    centre = out.loc["1", "geometry"]
    for p in lignes[lignes["id_station"] == "1"].geometry:
        assert centre.distance(p) < 7                        # les 2 points sont à ~10 m l'un de l'autre


def test_dates_ouverture_bounds(lignes):
    d = stations.dates_ouverture
    assert d(lignes, pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-31")) == [pd.Timestamp("2024-06-24")]
    assert d(lignes, pd.Timestamp("2024-06-24"), pd.Timestamp("2025-12-31")) == []   # borne basse exclue
    assert d(lignes, pd.Timestamp("2021-01-01"), pd.Timestamp("2024-06-24")) == [pd.Timestamp("2024-06-24")]


# ------------------------------------------------------------------- jointure
def test_dvf_to_gdf_projects_in_meters():
    df = pd.DataFrame({"longitude": [pt(0, 0)[0], pt(100, 0)[0]], "latitude": [pt(0, 0)[1]] * 2})
    g = stations.dvf_to_gdf(df)
    assert g.crs.to_epsg() == 2154 and len(g) == 2
    assert g.geometry.iloc[0].distance(g.geometry.iloc[1]) == pytest.approx(100, rel=0.02)


def test_labels():
    assert stations._labels([0, 100, 300, np.inf]) == ["0-100 m", "100-300 m", "300+ m"]


def make_dvf():
    xy = [pt(950, 0), pt(950, 0), pt(30, 0)]
    return pd.DataFrame({
        "longitude": [c[0] for c in xy],
        "latitude": [c[1] for c in xy],
        "date_mutation": pd.to_datetime(["2023-01-10", "2025-03-01", "2021-06-01"]),
        "prix_m2": [1.0, 2.0, 3.0],
    })


def test_network_depends_on_sale_date(lignes):
    """Même position (950 m d'Alpha, 50 m de Beta) : Beta n'existe qu'à partir de juin 2024."""
    out = stations.add_station_features(make_dvf(), lignes)
    assert out.loc[0, "nom_station"] == "Alpha"              # vente 2023 : Beta pas ouverte
    assert out.loc[0, "dist_station_m"] == pytest.approx(945, rel=0.03)
    assert out.loc[1, "nom_station"] == "Beta"               # vente 2025 : Beta ouverte
    assert out.loc[1, "dist_station_m"] == pytest.approx(50, rel=0.05)
    assert out.loc[1, "date_ouverture_station"] == pd.Timestamp("2024-06-24")
    assert out.loc[2, "nom_station"] == "Alpha"
    assert out.loc[2, "dist_station_m"] == pytest.approx(25, rel=0.05)


def test_columns_classes_and_neighbours(lignes):
    out = stations.add_station_features(make_dvf(), lignes)
    assert set(stations.STATION_COLS) <= set(out.columns)
    assert len(out) == 3 and out["prix_m2"].tolist() == [1.0, 2.0, 3.0]   # colonnes d'origine conservées
    assert out["classe_dist"].astype(str).tolist() == ["500+ m", "0-100 m", "0-100 m"]
    assert out["nb_stations_voisinage"].tolist() == [0, 1, 1]            # rayon de 500 m
    assert out.loc[0, "modes_station"] == "METRO+RER"


def test_custom_classes_follow_project_note(lignes):
    """Tranches de la note de cadrage : 0-100, 100-300, 300-500, >500 m."""
    out = stations.add_station_features(make_dvf(), lignes, classes=[0, 100, 300, 500, np.inf])
    assert out["classe_dist"].astype(str).tolist() == ["500+ m", "0-100 m", "0-100 m"]


def test_tied_stations_do_not_duplicate_sales():
    raw = gpd.GeoDataFrame(
        {"id_ref_zdc": [7, 8], "nom_zdc": ["P", "Q"], "res_com": ["METRO 1", "METRO 2"],
         "mode": ["METRO", "METRO"], "date_mise": pd.to_datetime([None, None], utc=True)},
        geometry=[Point(*pt(0, 0)), Point(*pt(0, 0))], crs="EPSG:4326",
    )
    lignes = stations.prepare_stations_lignes(raw, modes=["METRO"])
    dvf = pd.DataFrame({"longitude": [pt(40, 0)[0]], "latitude": [pt(40, 0)[1]],
                        "date_mutation": pd.to_datetime(["2023-01-01"])})
    out = stations.add_station_features(dvf, lignes)
    assert len(out) == 1                                     # deux stations à égale distance


def test_distances_and_neighbours_match_brute_force():
    """Compare au calcul direct sur 300 ventes : détecte tout décalage d'alignement."""
    rng = np.random.default_rng(0)
    n_st, n_v = 40, 300
    raw = gpd.GeoDataFrame(
        {"id_ref_zdc": np.arange(100, 100 + n_st), "nom_zdc": [f"S{i}" for i in range(n_st)],
         "res_com": "METRO 1", "mode": "METRO", "date_mise": pd.to_datetime([None] * n_st, utc=True)},
        geometry=[Point(*pt(x, y)) for x, y in rng.uniform(-3000, 3000, (n_st, 2))], crs="EPSG:4326",
    )
    lignes = stations.prepare_stations_lignes(raw, modes=["METRO"])
    vxy = [pt(x, y) for x, y in rng.uniform(-3000, 3000, (n_v, 2))]
    dvf = pd.DataFrame({"longitude": [c[0] for c in vxy], "latitude": [c[1] for c in vxy],
                        "date_mutation": pd.Timestamp("2023-01-01")})

    out = stations.add_station_features(dvf, lignes)

    ventes = stations.dvf_to_gdf(dvf)
    st = stations.stations_ouvertes(lignes, pd.Timestamp("2023-01-01"))
    v = np.c_[ventes.geometry.x, ventes.geometry.y]
    s = np.c_[st.geometry.x, st.geometry.y]
    d = np.sqrt(((v[:, None, :] - s[None, :, :]) ** 2).sum(-1))

    assert out["dist_station_m"].to_numpy() == pytest.approx(d.min(axis=1), rel=1e-6)
    np.testing.assert_array_equal(out["nb_stations_voisinage"].to_numpy(), (d <= 500).sum(axis=1))
    assert out["nom_station"].tolist() == st["nom_station"].to_numpy()[d.argmin(axis=1)].tolist()


# ---------------------------------------------------------- chargement du DVF
@pytest.fixture
def dossiers_vides(tmp_path, monkeypatch):
    """Isole charger_dvf_nettoye des vrais dossiers data/processed du projet."""
    monkeypatch.setattr(stations, "DOSSIER", tmp_path / "a")
    monkeypatch.setattr(stations, "RACINE_PROJET", tmp_path / "b")
    return tmp_path / "processed"


def test_charger_dvf_from_parquet(dossiers_vides):
    dossiers_vides.mkdir()
    pd.DataFrame({"date_mutation": pd.to_datetime(["2023-01-01"])}).to_parquet(
        dossiers_vides / "dvf_appartements.parquet")
    assert len(stations.charger_dvf_nettoye(dossiers_vides)) == 1


def test_charger_dvf_from_yearly_csv(dossiers_vides):
    dossiers_vides.mkdir()
    for an in (2021, 2022):
        pd.DataFrame({"date_mutation": [f"{an}-05-01"]}).to_csv(
            dossiers_vides / f"dvf_appartements_{an}.csv", index=False)
    assert len(stations.charger_dvf_nettoye(dossiers_vides)) == 2


def test_charger_dvf_raises_when_missing(dossiers_vides):
    with pytest.raises(FileNotFoundError):
        stations.charger_dvf_nettoye(dossiers_vides)

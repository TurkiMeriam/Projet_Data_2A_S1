"""Tests de la jointure stations <-> DVF sur un petit réseau fictif (Lambert 93)."""
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from vrai_prix import stations as st

X0, Y0 = 652_000, 6_862_000  # environ le centre de Paris en Lambert 93


def reseau_idfm() -> gpd.GeoDataFrame:
    """Format brut IDFM (WGS84, une ligne par gare x ligne, dates en texte)."""
    rows = [
        # id_ref_zdc, nom, res_com, mode, date_mise, x, y
        (1, "A", "METRO 1", "METRO", "1900-07-19T00:00:00+00:00", X0, Y0),
        (2, "B", "METRO 4", "METRO", "1910-01-09T00:00:00+00:00", X0 + 1000, Y0),
        (2, "B", "METRO 14", "METRO", "2024-06-24T00:00:00+00:00", X0 + 1010, Y0),
        (3, "C", "METRO 14", "METRO", "2024-06-24T00:00:00+00:00", X0, Y0 + 400),
        (4, "T", "TRAM 3a", "TRAMWAY", "2006-12-16T00:00:00+00:00", X0, Y0 + 50),
    ]
    df = pd.DataFrame(rows, columns=["id_ref_zdc", "nom_zdc", "res_com", "mode", "date_mise", "x", "y"])
    gdf = gpd.GeoDataFrame(df.drop(columns=["x", "y"]),
                           geometry=gpd.points_from_xy(df["x"], df["y"]), crs=2154)
    return gdf.to_crs(4326).assign(colonne_inutile=1)


def ventes() -> pd.DataFrame:
    pts = [
        ("s1", X0, Y0 + 100, "2023-01-01"),          # 100 m de A
        ("s2", X0, Y0 + 350, "2023-01-01"),          # A à 350 m, C pas encore ouverte
        ("s3", X0, Y0 + 350, "2025-01-01"),          # C ouverte, à 50 m
        ("s4", X0 + 1000, Y0 + 250, "2023-01-01"),   # B, une seule ligne
        ("s5", X0 + 1000, Y0 + 250, "2025-01-01"),   # B, deux lignes
    ]
    g = gpd.GeoSeries([Point(x, y) for _, x, y, _ in pts], crs=2154).to_crs(4326)
    return pd.DataFrame({
        "id_mutation": [p[0] for p in pts],
        "date_mutation": pd.to_datetime([p[3] for p in pts]),
        "longitude": g.x, "latitude": g.y,
    })


@pytest.fixture
def lignes():
    return st.prepare_stations_lignes(reseau_idfm())


def test_prepare_filtre_modes_et_projette(lignes):
    assert set(lignes["mode"]) == {"METRO"}          # tram exclu
    assert lignes.crs.to_epsg() == 2154
    assert lignes["date_mise"].dt.tz is None
    assert len(lignes) == 4


def test_stations_ouvertes_selon_la_date(lignes):
    avant = st.stations_ouvertes(lignes, pd.Timestamp("2023-01-01")).set_index("id_station")
    apres = st.stations_ouvertes(lignes, pd.Timestamp("2025-01-01")).set_index("id_station")
    assert set(avant.index) == {"1", "2"}
    assert set(apres.index) == {"1", "2", "3"}
    assert avant.loc["2", "nb_lignes_station"] == 1
    assert apres.loc["2", "nb_lignes_station"] == 2
    assert apres.loc["2", "lignes_station"] == "METRO 14, METRO 4"


def test_add_station_features(lignes):
    dvf = ventes()
    out = st.add_station_features(dvf, lignes).set_index("id_mutation")
    assert len(out) == len(dvf)

    assert out.loc["s1", "nom_station"] == "A"
    assert out.loc["s1", "dist_station_m"] == pytest.approx(100, abs=0.5)
    assert out.loc["s1", "classe_dist"] == "0-200 m"

    # Même logement, avant / après l'ouverture de la ligne 14
    assert out.loc["s2", "nom_station"] == "A"
    assert out.loc["s2", "dist_station_m"] == pytest.approx(350, abs=0.5)
    assert out.loc["s2", "nb_stations_voisinage"] == 1
    assert out.loc["s3", "nom_station"] == "C"
    assert out.loc["s3", "dist_station_m"] == pytest.approx(50, abs=0.5)
    assert out.loc["s3", "nb_stations_voisinage"] == 2

    assert out.loc["s4", "nb_lignes_station"] == 1
    assert out.loc["s4", "classe_dist"] == "200-400 m"
    assert out.loc["s5", "nb_lignes_station"] == 2


def test_read_geojson_roundtrip(tmp_path):
    p = tmp_path / "gares.geojson"
    reseau_idfm().to_file(p, driver="GeoJSON")
    lignes = st.read_stations_lignes(p)
    out = st.add_station_features(ventes(), lignes)
    assert out["dist_station_m"].notna().all()

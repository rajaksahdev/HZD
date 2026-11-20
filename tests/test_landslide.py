"""Stage 10: landslide susceptibility. Offline: synthetic cells and inventories."""

import numpy as np
import pandas as pd
import pytest

from models import landslide


def synthetic_cells(n_blocks: int = 10, per_block: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_blocks * per_block
    table = pd.DataFrame({name: rng.random(n) for name in landslide.FEATURE_SETS["with_roads"]})
    table["slope_p90_deg"] = rng.uniform(0, 45, n)
    table["block"] = np.repeat(np.arange(n_blocks), per_block)
    # landslides on the steepest cells, two in every block
    table["landslide"] = 0
    for b in range(n_blocks):
        idx = table.index[table["block"] == b]
        steepest = table.loc[idx, "slope_p90_deg"].nlargest(2).index
        table.loc[steepest, "landslide"] = 1
    return table


def test_blocks_reshape_keeps_each_cell_together():
    a = np.arange(36).reshape(6, 6)
    out = landslide._blocks(a, 3)
    assert out.shape == (2, 2, 9)
    assert sorted(out[0, 1]) == [3, 4, 5, 9, 10, 11, 15, 16, 17]


def test_cell_of_counts_rows_from_the_north_edge():
    west, south, east, north = landslide.REGION
    rows, cols = landslide.cell_of([north - 0.005, south + 0.005], [west + 0.005, east - 0.005])
    shape = landslide.grid_shape()
    assert (rows[0], cols[0]) == (0, 0)
    assert (rows[1], cols[1]) == (shape[0] - 1, shape[1] - 1)


def test_only_well_located_events_label_a_cell():
    table = pd.DataFrame({"row": [0, 0, 5], "col": [0, 1, 5]})
    west, _, _, north = landslide.REGION
    inventory = pd.DataFrame(
        {
            "lat": [north - 0.005, north - 0.005],
            "lon": [west + 0.005, west + 0.015],
            "accuracy": ["1km", "25km"],
        }
    )
    assert landslide.label(table, inventory)["landslide"].tolist() == [1, 0, 0]


def test_folds_never_split_a_block():
    table = synthetic_cells()
    fold = landslide.folds(table)
    assert pd.Series(fold).groupby(table["block"].to_numpy()).nunique().max() == 1
    assert set(fold) == set(range(landslide.FOLDS))


def test_recall_counts_landslides_in_the_top_share_of_each_fold():
    predictions = pd.DataFrame(
        {
            "fold": [0] * 10 + [1] * 10,
            "landslide": [1] + [0] * 9 + [1] + [0] * 9,
            "score": list(range(10, 0, -1)) + list(range(1, 11)),
        }
    )
    # fold 0: the landslide has the highest score; fold 1: the lowest
    assert landslide.recall_at(predictions, "score", 0.10) == (1, 2)


def test_held_out_scores_find_a_terrain_signal():
    table = synthetic_cells()
    predictions = landslide.held_out(table)
    scored = landslide.score(predictions).set_index("key")
    assert scored.loc["slope", "recall_10"] == pytest.approx(1.0)
    assert scored.loc["xgb_terrain", "pr_auc"] > table["landslide"].mean() * 3


def test_catalog_events_inside_the_region(monkeypatch, tmp_path):
    csv = tmp_path / "glc_x.csv"
    pd.DataFrame(
        {
            "event_date": ["07/30/2014 12:00:00 AM", "07/30/2014 12:00:00 AM"],
            "latitude": [19.162, 28.6],
            "longitude": [73.687, 77.2],
            "location_accuracy": ["1km", "exact"],
            "landslide_category": ["mudslide", "landslide"],
            "event_title": ["Malin", "Delhi"],
            "source_link": ["https://example.org/a", "https://example.org/b"],
        }
    ).to_csv(csv, index=False)
    monkeypatch.setattr(landslide, "glc_path", lambda: csv)
    monkeypatch.setattr(landslide, "RAW", tmp_path)
    table = landslide.events()
    assert table["title"].tolist() == ["Malin"]
    assert str(table.loc[0, "date"]) == "2014-07-30" and table.loc[0, "source"] == "NASA GLC"


def test_gsi_export_is_read_when_present(monkeypatch, tmp_path):
    (tmp_path / "gsi").mkdir()
    (tmp_path / "gsi" / "slides.geojson").write_text(
        '{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},'
        '"geometry":{"type":"Point","coordinates":[73.2,18.9]}}]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(landslide, "RAW", tmp_path)
    out = landslide.gsi_events()
    assert len(out) == 1 and out.loc[0, "source"] == "GSI"
    assert out.loc[0, "lat"] == pytest.approx(18.9)

"""Structural-contract test of the Gaborieau et al. paper repo.

Mirrors the column-building logic of `dev/predictions/predict_all_phages.py` from
the paper repo (https://github.com/mdmparis/coli_phage_interactions_2023) against
the published data, and asserts the structural claims documented in
`docs/tabula_research.md` Sec.6.

This is not a unit test of project code -- the project doesn't have implementation
code yet. It's a regression check against an external dataset: if the paper repo
ever changes shape (column renames, row additions, etc.), the assertions here
will fail and tell us exactly which doc claim went stale.

Run with: `./scripts/test.sh` (or `pytest tests/test_paper_pipeline_structure.py`).

Requirements:
- pandas (provided transitively by requirements-dev.txt).
- The paper repo cloned at `<project_root>/paper-repo/`. If missing, the test
  auto-clones, or skips with a clear message if cloning fails (e.g. no network).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# Doc-claimed values from docs/tabula_research.md Sec.6 and Sec.17.
# Update these alongside the doc if the paper repo's structure changes.
EXPECTED_INTERACTION_MATRIX_SHAPE = (402, 96)
EXPECTED_PICARD_ROWS = 403
EXPECTED_UMAP_DIMS = 8
EXPECTED_BACT_FEATURES_POST_FILTER = 13  # 8 UMAP + 5 categorical
EXPECTED_PRE_ONE_HOT_NORMAL = 18
EXPECTED_PRE_ONE_HOT_LF110 = 13
EXPECTED_POST_ONE_HOT_NORMAL = 114
EXPECTED_POST_ONE_HOT_LF110 = 109
EXPECTED_N_LF110_PHAGES = 4


@pytest.fixture(scope="module")
def loaded(paper_repo: Path) -> dict[str, pd.DataFrame]:
    """Load the four input files used by predict_all_phages.py.

    Returns:
        Dict with keys 'interaction_matrix', 'phage_features', 'bact_features_raw',
        'bact_embeddings'.
    """
    interaction_matrix = pd.read_csv(
        paper_repo / "data" / "interactions" / "interaction_matrix.csv", sep=";",
    ).set_index("bacteria")
    # Equivalent of the script's Phage_features_with_host.csv:
    phage_features = pd.read_csv(
        paper_repo / "data" / "genomics" / "phages" / "guelin_collection.csv", sep=";",
    ).set_index("phage").loc[interaction_matrix.columns, ["Morphotype", "Genus", "Phage_host"]]
    bact_features_raw = pd.read_csv(
        paper_repo / "data" / "genomics" / "bacteria" / "picard_collection.csv", sep=";",
    ).set_index("bacteria")
    bact_embeddings = pd.read_csv(
        paper_repo / "data" / "genomics" / "bacteria" / "umap_phylogeny" / "coli_umap_8_dims.tsv",
        sep="\t",
    ).set_index("bacteria")
    return {
        "interaction_matrix": interaction_matrix,
        "phage_features": phage_features,
        "bact_features_raw": bact_features_raw,
        "bact_embeddings": bact_embeddings,
    }


@pytest.fixture(scope="module")
def bact_features(loaded: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Bacterial features after the line-54 UMAP merge and line-57 regex filter.

    Mirrors `predict_all_phages.py` lines 54-57.
    """
    bf: pd.DataFrame = pd.merge(
        loaded["bact_features_raw"], loaded["bact_embeddings"],
        left_index=True, right_index=True,
    )
    return bf.filter(regex="(UMAP|O-type|LPS|ST_Warwick|Klebs|ABC_serotype)", axis=1)


def build_X_for_phage(
    p: str,
    interaction_matrix: pd.DataFrame,
    phage_features: pd.DataFrame,
    bact_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the pre-one-hot X and post-one-hot X_oh for a single phage.

    Faithful trace of `predict_all_phages.py` lines 59-119 (the per-phage loop), minus
    the CV split / model fit. Skips the cv_clusters merge since the grouping file is
    missing from the repo -- bacterium-as-its-own-group is fine for shape checks.
    """
    is_lf110 = p.startswith("LF110")

    # phage_feat reduced to just Phage_host (line 66)
    phage_feat = phage_features.loc[[p]].drop(["Morphotype", "Genus"], axis=1)
    interaction_mat = interaction_matrix[[p]]

    # wide -> long (line 69)
    iml = (
        interaction_mat.unstack().reset_index()
        .rename({"level_0": "phage", 0: "y"}, axis=1)
        .sort_values(["bacteria", "phage"])
    )
    # Fake the cv_clusters merge (cv groups file missing from the repo) (line 72)
    cv = pd.DataFrame({"group": iml["bacteria"].values}, index=iml["bacteria"].values)
    cv.index.name = "bacteria"
    cv = cv.drop_duplicates()
    iml = pd.merge(iml, cv, left_on=["bacteria"], right_index=True).set_index("group")

    # Merge bact_features (line 76)
    iwf = pd.merge(iml, bact_features, left_on=["bacteria"], right_index=True)

    # phage_host_features (line 79) -- includes phantom renames for cols already filtered out
    phf = pd.merge(
        phage_feat,
        bact_features.filter(regex="(ST_Warwick|O-type|H-type)", axis=1),
        left_on="Phage_host", right_index=True,
    ).rename({
        "Clermont_Phylo": "Clermont_host",
        "LPS_type": "LPS_host",
        "O-type": "O_host",
        "H-type": "H_host",
        "ST_Warwick": "ST_host",
    }, axis=1)

    # Merge phage_host_features (line 82), skipped for LF110
    if not is_lf110:
        iwf = pd.merge(iwf, phf.drop(["Phage_host"], axis=1), left_on="phage", right_index=True)

    # Rare-category binning + same_*_as_host engineering (lines 84-104)
    if "O-type" in bact_features.columns:
        otr = bact_features.groupby("O-type").filter(lambda x: x.shape[0] < 3)["O-type"].unique()
        iwf.loc[iwf["O-type"].isin(otr), "O-type"] = "Other"
        if not is_lf110:
            iwf["same_O_as_host"] = iwf["O-type"] == iwf["O_host"]
            iwf = iwf.drop("O_host", axis=1)
    if "ST_Warwick" in bact_features.columns:
        sr = bact_features.groupby("ST_Warwick").filter(lambda x: x.shape[0] < 3)["ST_Warwick"].unique()
        iwf.loc[iwf["ST_Warwick"].isin(sr), "ST_Warwick"] = "Other"
        if not is_lf110:
            iwf["same_ST_as_host"] = iwf["ST_Warwick"] == iwf["ST_host"]
    if "ABC_serotype" in bact_features.columns and not is_lf110:
        # The bug: column compared to itself; NaN==NaN is False in pandas
        iwf["same_ABC_as_host"] = iwf["ABC_serotype"] == iwf["ABC_serotype"]
    if "same_O_as_host" in iwf.columns and "same_ST_as_host" in iwf.columns and not is_lf110:
        iwf["same_O_and_ST_as_host"] = iwf["same_O_as_host"] * iwf["same_ST_as_host"]

    # Drop NaN-y rows (lines 107-108) and pull out X (line 111)
    iwf = iwf.dropna(subset=["y"])
    X = iwf.drop(["bacteria", "phage", "y"], axis=1)

    # One-hot encode (line 119)
    num = [c for c in X.columns if X.dtypes[c] == "float64"]
    fac = [c for c in X.columns if X.dtypes[c] != "float64"]
    X_oh = pd.concat([
        (X[num] - X[num].mean(axis=0)) / X[num].std(axis=0),
        pd.get_dummies(X[fac], sparse=False),
    ], axis=1)
    return X, X_oh


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_input_file_shapes(loaded: dict[str, pd.DataFrame]) -> None:
    """The four input files have the shapes the docs claim."""
    assert loaded["interaction_matrix"].shape == EXPECTED_INTERACTION_MATRIX_SHAPE, (
        f"interaction_matrix.csv: expected {EXPECTED_INTERACTION_MATRIX_SHAPE}, "
        f"got {loaded['interaction_matrix'].shape}"
    )
    assert loaded["bact_features_raw"].shape[0] == EXPECTED_PICARD_ROWS, (
        f"picard_collection.csv: expected {EXPECTED_PICARD_ROWS} rows, "
        f"got {loaded['bact_features_raw'].shape[0]}"
    )
    assert loaded["bact_embeddings"].shape[1] == EXPECTED_UMAP_DIMS, (
        f"coli_umap_8_dims.tsv: expected {EXPECTED_UMAP_DIMS} UMAP columns, "
        f"got {loaded['bact_embeddings'].shape[1]}"
    )


def test_402_vs_403_anomaly(loaded: dict[str, pd.DataFrame]) -> None:
    """One strain is in the catalog but missing from the interaction matrix.

    Documented in research.md Sec.2.4. This test pins the anomaly so a future
    repo fix (matrix grown to 403) surfaces immediately.
    """
    catalog_strains = set(loaded["bact_features_raw"].index)
    matrix_strains = set(loaded["interaction_matrix"].index)
    missing = catalog_strains - matrix_strains
    assert len(missing) == 1, (
        f"Expected exactly 1 strain in catalog but not in matrix; got {len(missing)}: {missing}"
    )


def test_regex_filter_drops_H_type(bact_features: pd.DataFrame) -> None:
    """The line-57 regex filter drops H-type before the line-79 rename runs.

    This is why H_host is a phantom: by the time the rename happens, H-type
    is already gone from bact_features. Documented in research.md Sec.6.2.
    """
    assert "H-type" not in bact_features.columns, (
        "H-type should have been dropped by the line-57 regex filter "
        "'(UMAP|O-type|LPS|ST_Warwick|Klebs|ABC_serotype)'"
    )
    assert bact_features.shape[1] == EXPECTED_BACT_FEATURES_POST_FILTER, (
        f"bact_features post-filter: expected {EXPECTED_BACT_FEATURES_POST_FILTER} cols, "
        f"got {bact_features.shape[1]}"
    )


def test_pre_one_hot_column_count_normal_phages(
    loaded: dict[str, pd.DataFrame], bact_features: pd.DataFrame,
) -> None:
    """All 92 non-LF110 phages produce exactly 18 pre-one-hot columns.

    Documented in research.md Sec.6.2 (pinned at 18 across the panel).
    """
    pre_counts: dict[str, int] = {}
    for p in loaded["phage_features"].index:
        if p.startswith("LF110"):
            continue
        X, _ = build_X_for_phage(p, loaded["interaction_matrix"], loaded["phage_features"], bact_features)
        pre_counts[p] = X.shape[1]
    unique = set(pre_counts.values())
    assert unique == {EXPECTED_PRE_ONE_HOT_NORMAL}, (
        f"Expected all non-LF110 phages to have {EXPECTED_PRE_ONE_HOT_NORMAL} pre-one-hot "
        f"columns; got unique values {unique}. Counts: {pre_counts}"
    )


def test_H_host_never_appears(
    loaded: dict[str, pd.DataFrame], bact_features: pd.DataFrame,
) -> None:
    """H_host is a phantom rename target; it should not appear in any phage's X."""
    for p in loaded["phage_features"].index:
        X, X_oh = build_X_for_phage(p, loaded["interaction_matrix"], loaded["phage_features"], bact_features)
        assert "H_host" not in X.columns, f"H_host appeared in X for phage {p}"
        h_host_dummies = [c for c in X_oh.columns if c.startswith("H_host")]
        assert not h_host_dummies, f"H_host one-hot dummies in X_oh for {p}: {h_host_dummies}"


def test_LF110_carve_out(
    loaded: dict[str, pd.DataFrame], bact_features: pd.DataFrame,
) -> None:
    """The 4 LF110 phages skip phage-host merge and same_*_as_host engineering."""
    lf110 = [p for p in loaded["phage_features"].index if p.startswith("LF110")]
    assert len(lf110) == EXPECTED_N_LF110_PHAGES, (
        f"Expected {EXPECTED_N_LF110_PHAGES} LF110 phages; got {len(lf110)}: {lf110}"
    )
    for p in lf110:
        X, _ = build_X_for_phage(p, loaded["interaction_matrix"], loaded["phage_features"], bact_features)
        assert X.shape[1] == EXPECTED_PRE_ONE_HOT_LF110, (
            f"LF110 phage {p}: expected {EXPECTED_PRE_ONE_HOT_LF110} pre-one-hot cols, got {X.shape[1]}"
        )
        for engineered in ("ST_host", "same_O_as_host", "same_ST_as_host", "same_ABC_as_host", "same_O_and_ST_as_host"):
            assert engineered not in X.columns, (
                f"LF110 phage {p} unexpectedly has engineered column {engineered}"
            )


def test_post_one_hot_dim(
    loaded: dict[str, pd.DataFrame], bact_features: pd.DataFrame,
) -> None:
    """Post-one-hot dim is 114 for all 92 normal phages and 109 for the 4 LF110 phages."""
    counts: dict[str, int] = {}
    for p in loaded["phage_features"].index:
        _, X_oh = build_X_for_phage(p, loaded["interaction_matrix"], loaded["phage_features"], bact_features)
        counts[p] = X_oh.shape[1]
    normal = {p: c for p, c in counts.items() if not p.startswith("LF110")}
    lf110 = {p: c for p, c in counts.items() if p.startswith("LF110")}
    assert set(normal.values()) == {EXPECTED_POST_ONE_HOT_NORMAL}, (
        f"Expected post-one-hot dim {EXPECTED_POST_ONE_HOT_NORMAL} for normal phages; "
        f"got unique values {set(normal.values())}"
    )
    assert set(lf110.values()) == {EXPECTED_POST_ONE_HOT_LF110}, (
        f"Expected post-one-hot dim {EXPECTED_POST_ONE_HOT_LF110} for LF110 phages; "
        f"got {lf110}"
    )


def test_same_ABC_bug_is_NaN_indicator_not_constant(
    loaded: dict[str, pd.DataFrame], bact_features: pd.DataFrame,
) -> None:
    """same_ABC_as_host is True iff ABC_serotype is non-NaN -- not always True.

    Documented in research.md Sec.6.3. The bug compares the column to itself, and
    in pandas `NaN == NaN` is False, so the column collapses to a presence indicator
    rather than a constant.
    """
    p = next(ph for ph in loaded["phage_features"].index if not ph.startswith("LF110"))

    interaction_mat = loaded["interaction_matrix"][[p]]
    iml = (
        interaction_mat.unstack().reset_index()
        .rename({"level_0": "phage", 0: "y"}, axis=1).sort_values(["bacteria", "phage"])
    )
    iwf = pd.merge(iml, bact_features, left_on=["bacteria"], right_index=True)
    iwf["same_ABC_as_host"] = iwf["ABC_serotype"] == iwf["ABC_serotype"]
    iwf = iwf.dropna(subset=["y"])

    n_nan = iwf["ABC_serotype"].isna().sum()
    n_true = iwf["same_ABC_as_host"].sum()
    n_false = (~iwf["same_ABC_as_host"]).sum()

    # Sanity: there ARE NaNs (most strains have no ABC capsule). If this fails the
    # repo's data has changed and the doc's "NaN-indicator" framing might need an update.
    assert n_nan > 0, f"Expected some NaN ABC_serotype values; got {n_nan}"
    # The bug doesn't yield "always True":
    assert n_false == n_nan, (
        f"Expected same_ABC_as_host False count ({n_false}) to equal NaN count ({n_nan})"
    )
    assert n_true == len(iwf) - n_nan, (
        f"Expected True count ({n_true}) to be (total - NaN) ({len(iwf) - n_nan})"
    )

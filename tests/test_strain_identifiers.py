"""Structural-contract tests for strain identifiers across the paper-repo files.

The 402 strains we want to predict on appear under several naming conventions
in the repo. After investigation, the cleanest way to look up phylogenetic
distances by bacteria name is via the *_orignames distance matrix:

- `interaction_matrix.csv` -- friendly bacteria names like `ECOR-54`
  (402 rows, the working dataset).
- `picard_collection.csv` -- friendly bacteria names + a `Gembase` column
  with IDs like `ESCO.0622.00024`, 403 rows. **The Gembase column is NOT
  the bridge to the distance matrix** -- it's a different identifier scheme
  (database registration codes), not the PanACoTA build's row IDs.
- `370+host_distance_matrix.tsv` -- 404x404, indexed by sequential PanACoTA
  IDs `ESCO.0722.00001` through `ESCO.0722.00404`. Same matrix as the
  orignames version below, just labelled with the build IDs.
- `370+host_distance_matrix_orignames.tsv` -- the same 404x404 matrix but
  indexed by friendly bacteria names. This is the version we should use
  for code; no name-mapping required.
- `strain_name_gembase_correspondance_all.csv` -- explicit ESCO.0722.NNNNN
  to bacteria-name mapping. Useful for cross-checks but not needed for CV.

These tests pin all of that. If the paper repo ever changes shape, this file
fails first, before any cross-validation or featurization code that depends
on the mapping silently breaks.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# Pinned from inspection of the published files.
EXPECTED_INTERACTION_STRAINS = 402
EXPECTED_PICARD_ROWS = 403
EXPECTED_DISTANCE_MATRIX_SIZE = 404  # square; same in both index variants

DISTANCE_MATRIX_ORIGNAMES_RELPATH = (
    "data/genomics/bacteria/isolation_strains/panacota/tree/370+host_distance_matrix_orignames.tsv"
)
DISTANCE_MATRIX_GEMBASE_RELPATH = (
    "data/genomics/bacteria/isolation_strains/panacota/tree/370+host_distance_matrix.tsv"
)
CORRESPONDANCE_RELPATH = (
    "data/genomics/bacteria/isolation_strains/panacota/strain_name_gembase_correspondance_all.csv"
)


@pytest.fixture(scope="module")
def interaction_strains(paper_repo: Path) -> pd.Index:
    """The 402 bacteria names that have interaction labels."""
    df = pd.read_csv(paper_repo / "data/interactions/interaction_matrix.csv", sep=";")
    return df.set_index("bacteria").index


@pytest.fixture(scope="module")
def picard(paper_repo: Path) -> pd.DataFrame:
    """The 403-row strain catalog, indexed by friendly bacteria name."""
    return pd.read_csv(
        paper_repo / "data/genomics/bacteria/picard_collection.csv", sep=";"
    ).set_index("bacteria")


@pytest.fixture(scope="module")
def distance_matrix_orignames(paper_repo: Path) -> pd.DataFrame:
    """The 404x404 pairwise phylogenetic distance matrix, indexed by friendly bacteria
    names. This is the primary file CV code should use -- no name mapping required."""
    return pd.read_csv(paper_repo / DISTANCE_MATRIX_ORIGNAMES_RELPATH, sep="\t").set_index("bacteria")


@pytest.fixture(scope="module")
def distance_matrix_gembase(paper_repo: Path) -> pd.DataFrame:
    """The same 404x404 matrix as above, indexed by PanACoTA-build IDs (ESCO.0722.NNNNN).
    Kept around for cross-checking that the two versions agree numerically."""
    return pd.read_csv(paper_repo / DISTANCE_MATRIX_GEMBASE_RELPATH, sep="\t").set_index("bacteria")


@pytest.fixture(scope="module")
def correspondance(paper_repo: Path) -> pd.DataFrame:
    """Explicit Gembase <-> bacteria-name mapping. Whitespace-separated."""
    return pd.read_csv(
        paper_repo / CORRESPONDANCE_RELPATH, sep=r"\s+", engine="python"
    ).set_index("Gembase")


# ---------------------------------------------------------------------------
# Shape / size sanity
# ---------------------------------------------------------------------------

def test_file_sizes_as_expected(
    interaction_strains: pd.Index,
    picard: pd.DataFrame,
    distance_matrix_orignames: pd.DataFrame,
    distance_matrix_gembase: pd.DataFrame,
) -> None:
    """The four files have the row/column counts we expect."""
    assert len(interaction_strains) == EXPECTED_INTERACTION_STRAINS, (
        f"interaction_matrix.csv: expected {EXPECTED_INTERACTION_STRAINS} strains, "
        f"got {len(interaction_strains)}"
    )
    assert len(picard) == EXPECTED_PICARD_ROWS, (
        f"picard_collection.csv: expected {EXPECTED_PICARD_ROWS} rows, got {len(picard)}"
    )
    expected_shape = (EXPECTED_DISTANCE_MATRIX_SIZE, EXPECTED_DISTANCE_MATRIX_SIZE)
    assert distance_matrix_orignames.shape == expected_shape, (
        f"orignames distance matrix: expected {expected_shape}, got {distance_matrix_orignames.shape}"
    )
    assert distance_matrix_gembase.shape == expected_shape, (
        f"gembase distance matrix: expected {expected_shape}, got {distance_matrix_gembase.shape}"
    )


def test_distance_matrix_is_square(distance_matrix_orignames: pd.DataFrame) -> None:
    """Row index matches column index."""
    assert list(distance_matrix_orignames.index) == list(distance_matrix_orignames.columns)


# ---------------------------------------------------------------------------
# The orignames matrix is the bridge: every interaction strain is in its index
# ---------------------------------------------------------------------------

def test_every_interaction_strain_in_orignames_matrix(
    interaction_strains: pd.Index, distance_matrix_orignames: pd.DataFrame,
) -> None:
    """Every interaction-matrix strain must appear as a row in the orignames distance matrix."""
    matrix_strains = set(distance_matrix_orignames.index)
    missing = set(interaction_strains) - matrix_strains
    assert not missing, (
        f"{len(missing)} interaction-matrix strain(s) missing from orignames matrix: "
        f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
    )


def test_self_distance_is_zero_for_every_strain(
    distance_matrix_orignames: pd.DataFrame,
) -> None:
    """For every strain in the orignames matrix, distance to itself is exactly 0.0."""
    diag = pd.Series(
        [distance_matrix_orignames.loc[s, s] for s in distance_matrix_orignames.index],
        index=distance_matrix_orignames.index,
    )
    nonzero = diag[diag != 0.0]
    assert nonzero.empty, f"{len(nonzero)} strain(s) have nonzero self-distance: {nonzero.head().to_dict()}"


def test_distance_matrix_symmetric(distance_matrix_orignames: pd.DataFrame) -> None:
    """d(a, b) == d(b, a) for all strain pairs."""
    arr = distance_matrix_orignames.to_numpy()
    asymmetry = (arr != arr.T).sum()
    assert asymmetry == 0, f"Distance matrix is not symmetric ({asymmetry} cells differ from transpose)"


def test_self_distance_equals_row_minimum(distance_matrix_orignames: pd.DataFrame) -> None:
    """For every strain X, d(X, X) equals the minimum value in row X.

    A well-formed distance matrix has d(X, X) = 0 and no other strain has smaller
    distance to X than X itself. Ties are allowed: two distinct strains can have
    distance 0 if they're perfect clones, in which case the row min is 0 and the
    self entry equals that min.

    Stronger than diagonal-is-zero alone: catches e.g. a sign-flip that turned
    the matrix into a similarity matrix while leaving the diagonal at 0.

    Avoids `argmin` (which would break asymmetrically on clone pairs by tie-
    breaking via index order); compares row min to diagonal value directly so
    clones both pass.
    """
    arr = distance_matrix_orignames.to_numpy()
    strains = list(distance_matrix_orignames.index)
    row_mins = arr.min(axis=1)
    diagonals = arr.diagonal()
    bad = [
        (strains[i], float(diagonals[i]), float(row_mins[i]))
        for i in range(len(strains))
        if diagonals[i] != row_mins[i]
    ]
    assert not bad, (
        f"{len(bad)} strain(s) have self-distance differing from row minimum "
        f"(sample: {bad[:3]})"
    )


def test_known_strain_lookup_works(
    interaction_strains: pd.Index, distance_matrix_orignames: pd.DataFrame,
) -> None:
    """ECOR-54 specifically: in interaction_strains, in distance matrix, self-distance 0."""
    name = "ECOR-54"
    assert name in interaction_strains
    assert name in distance_matrix_orignames.index
    assert distance_matrix_orignames.loc[name, name] == 0.0


# ---------------------------------------------------------------------------
# The orignames vs gembase matrices should be numerically identical
# ---------------------------------------------------------------------------

def test_orignames_and_gembase_matrices_agree_numerically(
    distance_matrix_orignames: pd.DataFrame,
    distance_matrix_gembase: pd.DataFrame,
    correspondance: pd.DataFrame,
) -> None:
    """The two distance matrix versions are the same data with different labels;
    after relabelling the gembase version via the correspondance, the values must match."""
    gembase_relabeled = distance_matrix_gembase.rename(
        index=correspondance["Strain"].to_dict(),
        columns=correspondance["Strain"].to_dict(),
    )
    # Reindex both to the orignames order so we compare like with like.
    common = list(distance_matrix_orignames.index)
    a = distance_matrix_orignames.loc[common, common].to_numpy()
    b = gembase_relabeled.loc[common, common].to_numpy()
    import numpy as np
    assert np.allclose(a, b, atol=1e-9), (
        f"orignames and gembase distance matrices disagree numerically (max diff: "
        f"{np.abs(a - b).max()})"
    )


# ---------------------------------------------------------------------------
# Picard Gembase column is NOT the bridge -- it's a different identifier scheme
# ---------------------------------------------------------------------------

def test_picard_gembase_column_is_only_partial_bridge_to_distance_matrix(
    picard: pd.DataFrame, distance_matrix_gembase: pd.DataFrame,
) -> None:
    """Picard's Gembase column has only partial overlap with the distance matrix IDs.

    Most Picard strains were registered in June 2022 (`.0622.` month code) while
    the distance matrix uses July 2022 PanACoTA build IDs (`.0722.`). Only the
    ~33 strains whose Gembase happens to use `.0722.` overlap.

    Pin both halves: the overlap is non-empty (don't claim "no overlap" in docs)
    but is far below total (so the column is NOT usable as the bridge). Use the
    orignames distance matrix instead.
    """
    picard_gembases = set(picard["Gembase"].dropna())
    matrix_ids = set(distance_matrix_gembase.index)
    overlap = picard_gembases & matrix_ids
    n_picard = len(picard_gembases)
    n_overlap = len(overlap)
    assert n_overlap > 0, "Expected partial overlap (phage-isolation strains), got 0"
    assert n_overlap < n_picard * 0.2, (
        f"Picard Gembase column now overlaps with {n_overlap}/{n_picard} distance-matrix IDs. "
        f"If this is approaching full overlap, the docs and CV code should be updated to use "
        f"the Gembase column directly as the bridge."
    )


# ---------------------------------------------------------------------------
# Documented asymmetries
# ---------------------------------------------------------------------------

def test_one_picard_strain_missing_from_interaction_matrix(
    interaction_strains: pd.Index, picard: pd.DataFrame,
) -> None:
    """Picard has exactly one strain with metadata but no interaction labels.
    Documented in docs/tabula_research.md Sec.2.4 as the 402-vs-403 anomaly."""
    missing = set(picard.index) - set(interaction_strains)
    assert len(missing) == 1, (
        f"Expected exactly 1 Picard strain missing from interaction matrix; got {len(missing)}: "
        f"{sorted(missing)}"
    )


def test_distance_matrix_has_two_specific_orphans(
    distance_matrix_orignames: pd.DataFrame,
    interaction_strains: pd.Index,
    picard: pd.DataFrame,
) -> None:
    """Distance matrix has 404 strains, interaction matrix has 402. The two extras
    are `H1-005-0065-L-P` and `H27` -- in the distance matrix but in NEITHER picard
    NOR the interaction matrix. Per `dev/predictions/visualize_predictions.ipynb`
    cell 2 in the paper repo, these are "lost bacteria": strains that originally
    had interaction data measured but were dropped before the final analysis. The
    notebook sets `interaction_matrix.loc["H1-005-0065-L-P"] = np.nan` and same
    for H27 to align dimensions for visualization.

    Pinning the specific names: if either ever disappears or changes, this fires
    rather than letting downstream code silently re-tune to a different set."""
    matrix = set(distance_matrix_orignames.index)
    expected_orphans = {"H1-005-0065-L-P", "H27"}

    distance_only = matrix - set(interaction_strains) - set(picard.index)
    assert distance_only == expected_orphans, (
        f"Distance-matrix-only orphans changed. Expected {sorted(expected_orphans)}, "
        f"got {sorted(distance_only)}"
    )


def test_orphan_sets_are_disjoint(
    distance_matrix_orignames: pd.DataFrame,
    interaction_strains: pd.Index,
    picard: pd.DataFrame,
) -> None:
    """The strain that's in picard but not interactions (LF110) is a DIFFERENT strain
    from the two that are in the distance matrix but not picard. The orphan sets
    don't overlap. Pinning this is worth it because the docs got it wrong on the
    first pass ("the two distance extras likely include LF110") -- they don't."""
    picard_only = set(picard.index) - set(interaction_strains)
    distance_only = set(distance_matrix_orignames.index) - set(interaction_strains) - set(picard.index)
    overlap = picard_only & distance_only
    assert not overlap, (
        f"Orphan sets unexpectedly overlap: {sorted(overlap)}. "
        f"picard-only={sorted(picard_only)}, distance-only={sorted(distance_only)}"
    )


# ---------------------------------------------------------------------------
# Perfect-clone pairs in the distance matrix
# ---------------------------------------------------------------------------

# (a, b) pairs with d(a, b) == 0 but a != b. Listed alphabetically within each pair
# and across pairs for stable comparison. Two of these are true duplicates (rows are
# byte-identical against the rest of the matrix); one (IAI15/IAI17) has matching core
# genomes but rows differ at the ~4e-6 level, consistent with independent PanACoTA
# runs of nearly-identical strains.
EXPECTED_CLONE_PAIRS: list[tuple[str, str]] = [
    ("H1-001-0020-M-O", "H1-003-0090-V-J"),  # true duplicate (row-identical)
    ("IAI15", "IAI17"),                       # near-duplicate (independent runs)
    ("ROAR047", "ROAR072"),                   # true duplicate (row-identical)
]
EXPECTED_TRUE_DUPLICATE_PAIRS = {("H1-001-0020-M-O", "H1-003-0090-V-J"), ("ROAR047", "ROAR072")}


def _zero_distance_pairs(matrix: pd.DataFrame) -> list[tuple[str, str]]:
    """All (a, b) with d(a, b) == 0 and a < b lexicographically."""
    arr = matrix.to_numpy()
    strains = list(matrix.index)
    pairs: list[tuple[str, str]] = []
    for i in range(arr.shape[0]):
        for j in range(i + 1, arr.shape[0]):
            if arr[i, j] == 0.0:
                a, b = sorted([strains[i], strains[j]])
                pairs.append((a, b))
    return sorted(pairs)


def test_clone_pairs_are_exactly_the_expected_three(
    distance_matrix_orignames: pd.DataFrame,
) -> None:
    """Exactly these 3 strain pairs have zero off-diagonal distance. Pinning the
    specific pairs by name so a future repo update that introduces or removes
    clones surfaces here rather than silently changing CV behavior downstream."""
    observed = _zero_distance_pairs(distance_matrix_orignames)
    assert observed == EXPECTED_CLONE_PAIRS, (
        f"Clone pairs changed. Expected {EXPECTED_CLONE_PAIRS}, got {observed}"
    )


def test_true_duplicate_pairs_have_byte_identical_rows(
    distance_matrix_orignames: pd.DataFrame,
) -> None:
    """For the two pairs that are true duplicates (same biological strain registered
    under two names), every row entry matches exactly. The third pair (IAI15/IAI17)
    deliberately not asserted here -- it has matching core genomes but ~4e-6 numerical
    differences from independent PanACoTA runs."""
    for a, b in EXPECTED_TRUE_DUPLICATE_PAIRS:
        row_a = distance_matrix_orignames.loc[a].to_numpy()
        row_b = distance_matrix_orignames.loc[b].to_numpy()
        n_diff = int((row_a != row_b).sum())
        assert n_diff == 0, (
            f"Pair ({a}, {b}) expected to be byte-identical but {n_diff} cells differ"
        )


def test_iai15_iai17_pair_is_near_duplicate_not_exact(
    distance_matrix_orignames: pd.DataFrame,
) -> None:
    """The IAI15/IAI17 pair has distance(IAI15, IAI17) == 0 but their rows against
    the rest of the matrix differ slightly. This is consistent with two independent
    PanACoTA runs of strains that have identical core genomes -- the biology says
    "same", but the numerical pipeline produced different floating-point realizations.

    Pin both halves: distance is exactly 0, and rows are NOT byte-identical (so
    we don't claim more identity than the data shows).
    """
    a, b = "IAI15", "IAI17"
    assert distance_matrix_orignames.loc[a, b] == 0.0, (
        f"d({a}, {b}) should be 0.0; got {distance_matrix_orignames.loc[a, b]}"
    )
    row_a = distance_matrix_orignames.loc[a].to_numpy()
    row_b = distance_matrix_orignames.loc[b].to_numpy()
    import numpy as np
    n_diff = int((row_a != row_b).sum())
    max_diff = float(np.abs(row_a - row_b).max())
    assert n_diff > 0, "IAI15/IAI17 rows are byte-identical; reclassify as true duplicate"
    assert max_diff < 1e-5, (
        f"IAI15/IAI17 rows differ by more than expected (max abs diff {max_diff:.2e}); "
        f"the 'near-duplicate' framing may no longer hold"
    )

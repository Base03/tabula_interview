"""Tests for src.cv: grouped CV from the phylogenetic distance matrix.

Verifies:
  - Loading mechanics (all 402 strains have group labels, no NaN).
  - Necessary condition: every pair with direct d < threshold shares a group.
  - Sufficient condition: groups match connected components of the thresholded graph.
  - Effective group count is in the range the paper claims (250-350).
  - All 3 known clone pairs end up in the same group.
  - GroupKFold mechanics: same-group strains never split across train/test;
    every strain is in exactly one test fold across the 10-fold split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupKFold

from src.cv import DEFAULT_THRESHOLD, load_strain_groups

EXPECTED_N_STRAINS = 402
# Paper claims ~250-350 effective independent strains after grouping at 1e-4.
EXPECTED_N_GROUPS_MIN = 200
EXPECTED_N_GROUPS_MAX = 400

# Pinned in tests/test_strain_identifiers.py
EXPECTED_CLONE_PAIRS: list[tuple[str, str]] = [
    ("H1-001-0020-M-O", "H1-003-0090-V-J"),
    ("IAI15", "IAI17"),
    ("ROAR047", "ROAR072"),
]


@pytest.fixture(scope="module")
def groups(paper_repo: Path) -> pd.Series:
    return load_strain_groups(paper_repo)


@pytest.fixture(scope="module")
def distance_submatrix(paper_repo: Path) -> pd.DataFrame:
    """The 402 x 402 distance submatrix that load_strain_groups operates on."""
    interaction_matrix = pd.read_csv(
        paper_repo / "data/interactions/interaction_matrix.csv", sep=";"
    ).set_index("bacteria")
    distance_matrix = pd.read_csv(
        paper_repo
        / "data/genomics/bacteria/isolation_strains/panacota/tree/370+host_distance_matrix_orignames.tsv",
        sep="\t",
    ).set_index("bacteria")
    strains = list(interaction_matrix.index)
    return distance_matrix.loc[strains, strains]


# ---------------------------------------------------------------------------
# Loading mechanics
# ---------------------------------------------------------------------------

def test_all_strains_have_a_group(groups: pd.Series) -> None:
    """Every strain in the interaction matrix gets a group label, no NaN."""
    assert len(groups) == EXPECTED_N_STRAINS, (
        f"Expected {EXPECTED_N_STRAINS} grouped strains, got {len(groups)}"
    )
    assert groups.notna().all(), (
        f"Some strains have NaN group labels: {groups[groups.isna()].index.tolist()[:5]}"
    )


def test_groups_indexed_by_bacteria_name(groups: pd.Series) -> None:
    """The returned Series is indexed by bacteria name, matching interaction_matrix order."""
    assert groups.index.name == "bacteria"
    assert all(isinstance(s, str) for s in groups.index)


def test_group_labels_are_dense_integers(groups: pd.Series) -> None:
    """Group IDs are integers in [0, n_groups) -- not e.g. sparse or string labels."""
    assert pd.api.types.is_integer_dtype(groups), (
        f"Expected integer dtype for group labels, got {groups.dtype}"
    )
    n_groups = groups.nunique()
    assert set(groups.unique()) == set(range(n_groups)), (
        f"Group labels not dense in [0, {n_groups}); got {sorted(set(groups.unique()))[:5]}..."
    )


# ---------------------------------------------------------------------------
# Group count in expected range
# ---------------------------------------------------------------------------

def test_effective_n_in_expected_range(groups: pd.Series) -> None:
    """Paper claims ~250-350 effective independent strains after grouping at 1e-4.

    We assert a slightly wider range [200, 400] to allow for reconstruction differences
    (e.g. the paper may have used a slightly different threshold or different
    distance source). If this fails, investigate whether the threshold is right or
    the input distance matrix is the version the paper used.
    """
    n_groups = groups.nunique()
    assert EXPECTED_N_GROUPS_MIN <= n_groups <= EXPECTED_N_GROUPS_MAX, (
        f"Number of groups ({n_groups}) outside expected range "
        f"[{EXPECTED_N_GROUPS_MIN}, {EXPECTED_N_GROUPS_MAX}]; paper claims ~250-350"
    )


# ---------------------------------------------------------------------------
# Correctness: necessary and sufficient conditions
# ---------------------------------------------------------------------------

def test_necessary_condition_pairwise_near_clones_in_same_group(
    groups: pd.Series, distance_submatrix: pd.DataFrame,
) -> None:
    """Necessary condition: for every pair (i, j) with direct d < threshold,
    group(i) == group(j). See docs/tabula_research.md Sec.7.3 for the distinction
    between necessary (direct pairwise) and sufficient (transitive closure)."""
    strains = list(groups.index)
    dist = distance_submatrix.to_numpy()
    n = dist.shape[0]
    violations: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] < DEFAULT_THRESHOLD:
                if groups.iloc[i] != groups.iloc[j]:
                    violations.append((strains[i], strains[j], float(dist[i, j])))
    assert not violations, (
        f"{len(violations)} near-clone pair(s) ended up in different groups "
        f"(sample: {violations[:3]})"
    )


def test_sufficient_condition_groups_are_connected_components(
    groups: pd.Series, distance_submatrix: pd.DataFrame,
) -> None:
    """Sufficient condition: group assignments are exactly the connected components
    of the thresholded graph (the actual paper rule, including transitive closure).

    Group integers may be permuted vs ours, so we compare partitions (unordered sets
    of equivalence classes), not raw labels.
    """
    dist = distance_submatrix.to_numpy()
    adj = (dist < DEFAULT_THRESHOLD).astype(np.int8)
    _, expected_labels = connected_components(csr_matrix(adj), directed=False)

    def partition(labels: np.ndarray) -> frozenset[frozenset[int]]:
        df = pd.DataFrame({"label": labels})
        return frozenset(
            frozenset(grp.index.tolist())
            for _, grp in df.groupby("label")
        )

    observed = partition(np.asarray(groups.values, dtype=int))
    expected = partition(expected_labels)
    assert observed == expected, (
        "Group assignments do not match connected components of the thresholded graph"
    )


# ---------------------------------------------------------------------------
# Known clone pairs share a group
# ---------------------------------------------------------------------------

def test_known_clone_pairs_share_a_group(groups: pd.Series) -> None:
    """The 3 perfect-clone pairs pinned in test_strain_identifiers.py must end up in
    the same group. They're at distance exactly 0.0, well below threshold."""
    for a, b in EXPECTED_CLONE_PAIRS:
        assert groups.loc[a] == groups.loc[b], (
            f"Clone pair ({a}, {b}) ended up in different groups "
            f"({groups.loc[a]} vs {groups.loc[b]})"
        )


# ---------------------------------------------------------------------------
# GroupKFold mechanics
# ---------------------------------------------------------------------------

def test_GroupKFold_never_splits_a_group(groups: pd.Series) -> None:
    """sklearn's GroupKFold ensures same-group strains end up in the same fold
    (train xor test, never both). Verifying we can use our groups output correctly
    with the canonical paper API."""
    gkf = GroupKFold(n_splits=10)
    X = pd.DataFrame(index=groups.index)
    for train_idx, test_idx in gkf.split(X, groups=groups.to_numpy()):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        overlap = train_groups & test_groups
        assert not overlap, (
            f"GroupKFold violated: groups {sorted(overlap)} appear in both train and test"
        )


def test_every_strain_in_exactly_one_test_fold(groups: pd.Series) -> None:
    """Across 10 folds, each strain is in exactly one test set -- so concatenated
    out-of-fold predictions cover the panel exactly once. This is the property the
    paper relies on for its aggregated AUROC."""
    gkf = GroupKFold(n_splits=10)
    X = pd.DataFrame(index=groups.index)
    test_counts = np.zeros(len(groups), dtype=int)
    for _, test_idx in gkf.split(X, groups=groups.to_numpy()):
        test_counts[test_idx] += 1
    not_exactly_one = (test_counts != 1).sum()
    assert not_exactly_one == 0, (
        f"{not_exactly_one} strain(s) appeared in != 1 test folds (counts: {set(test_counts.tolist())})"
    )


# ---------------------------------------------------------------------------
# Reconstruction outputs pinned
# ---------------------------------------------------------------------------

# Filled in after first observation; if the upstream distance matrix changes,
# the threshold changes, or our algorithm changes, these values will diverge
# and the test will fire with the new numbers. At that point, decide whether to:
#   (a) update these constants (and the corresponding finding in §0)
#   (b) investigate why the reconstruction shifted.
EXPECTED_N_GROUPS_PINNED: int = 301
EXPECTED_LARGEST_GROUP_SIZE: int = 8
EXPECTED_N_SINGLETON_GROUPS: int = 244
EXPECTED_N_NONSINGLETON_GROUPS: int = 57


def test_group_reconstruction_outputs_pinned(groups: pd.Series) -> None:
    """Pin the specific reconstruction outputs from the published distance matrix
    at the 1e-4 threshold. If any of these shift, decide whether the new values
    are correct (and update them + the finding in §0) or whether something broke.
    """
    size_per_group = groups.value_counts()
    n_groups = int(groups.nunique())
    largest = int(size_per_group.max())
    n_singletons = int((size_per_group == 1).sum())
    n_nonsingletons = int((size_per_group >= 2).sum())

    failures: list[str] = []
    if EXPECTED_N_GROUPS_PINNED != -1 and n_groups != EXPECTED_N_GROUPS_PINNED:
        failures.append(f"n_groups: got {n_groups}, pinned {EXPECTED_N_GROUPS_PINNED}")
    if EXPECTED_LARGEST_GROUP_SIZE != -1 and largest != EXPECTED_LARGEST_GROUP_SIZE:
        failures.append(f"largest_group_size: got {largest}, pinned {EXPECTED_LARGEST_GROUP_SIZE}")
    if EXPECTED_N_SINGLETON_GROUPS != -1 and n_singletons != EXPECTED_N_SINGLETON_GROUPS:
        failures.append(f"n_singleton_groups: got {n_singletons}, pinned {EXPECTED_N_SINGLETON_GROUPS}")
    if EXPECTED_N_NONSINGLETON_GROUPS != -1 and n_nonsingletons != EXPECTED_N_NONSINGLETON_GROUPS:
        failures.append(f"n_nonsingleton_groups: got {n_nonsingletons}, pinned {EXPECTED_N_NONSINGLETON_GROUPS}")

    # If all pinned constants are still -1 (first run), fail with the observed values
    # so they can be filled in.
    if all(c == -1 for c in [
        EXPECTED_N_GROUPS_PINNED, EXPECTED_LARGEST_GROUP_SIZE,
        EXPECTED_N_SINGLETON_GROUPS, EXPECTED_N_NONSINGLETON_GROUPS,
    ]):
        pytest.fail(
            f"Pinning constants are placeholders. Observed values:\n"
            f"  EXPECTED_N_GROUPS_PINNED = {n_groups}\n"
            f"  EXPECTED_LARGEST_GROUP_SIZE = {largest}\n"
            f"  EXPECTED_N_SINGLETON_GROUPS = {n_singletons}\n"
            f"  EXPECTED_N_NONSINGLETON_GROUPS = {n_nonsingletons}\n"
            f"Fill these in at the top of this test file."
        )

    assert not failures, "Reconstruction shifted: " + "; ".join(failures)

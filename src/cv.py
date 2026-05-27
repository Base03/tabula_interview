"""Grouped cross-validation for the phage-host prediction task.

The paper uses 10-fold GroupKFold with groups defined by phylogenetic near-clones:
strains in the same connected component of the distance graph thresholded at
1e-4 substitutions/site go in the same fold. See `docs/tabula_research.md` Sec.7.3
for the transitive-closure subtlety.

This module reconstructs the {bacterium: group_id} mapping from the shipped
orignames distance matrix. The paper's `predict_all_phages.py:50` loaded this
mapping from `370+host_cross_validation_groups_1e-4.csv`, which is missing from
the repo (README:145). Downstream code (training.py, etc.) consumes the output
the same way the paper script did: by indexing into the returned Series and
passing the per-sample group array to `sklearn.model_selection.GroupKFold.split`.

Background notes:

- **Reconstruction != exact reproduction.** GroupKFold's specific fold assignments
  depend on group-integer ordering and sample order. Even with structurally correct
  groupings, our integer labels may differ from the paper's originals, and so the
  specific strain-to-fold mapping will differ too. Expect AUROC distributions to
  match within tolerance, not exact fold membership.
- **Why connected components, not pairwise thresholding.** The paper rule is
  transitive closure: if A-B and B-C are both near-clones (each < 1e-4) but A-C is
  not, A and C still end up in the same group via B. `connected_components` gives
  us this for free; naive pairwise grouping would silently miss it.
- **Why the orignames distance matrix specifically.** The `_orignames` variant is
  indexed by bacteria names directly (e.g. `ECOR-54`); the non-orignames version
  uses sequential PanACoTA build IDs (`ESCO.0722.NNNNN`) that don't match
  picard's `Gembase` column. See `tests/test_strain_identifiers.py` for the
  structural-contract test pinning this choice.
- **Lost bacteria.** The distance matrix has 404 rows but only 402 of them appear
  in `interaction_matrix.csv` -- `H1-005-0065-L-P` and `H27` are "lost bacteria"
  per `dev/predictions/visualize_predictions.ipynb` cell 2, strains dropped before
  the final analysis. We implicitly drop them by subsetting to the interaction
  matrix's strain index.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

DISTANCE_MATRIX_ORIGNAMES_RELPATH = (
    "data/genomics/bacteria/isolation_strains/panacota/tree/370+host_distance_matrix_orignames.tsv"
)
INTERACTION_MATRIX_RELPATH = "data/interactions/interaction_matrix.csv"

# The paper's threshold: strains within this core-genome distance go in the same fold.
DEFAULT_THRESHOLD: float = 1e-4


def load_strain_groups(
    paper_repo: Path,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.Series:
    """Compute connected-component group labels for the interaction-matrix strains.

    Algorithm:
      1. Load the orignames distance matrix (404 x 404, indexed by bacteria names).
      2. Load the interaction matrix to determine the 402 strains we care about.
      3. Subset the distance matrix to those 402 strains, preserving interaction-matrix
         order. This implicitly drops the 2 "lost bacteria" not in the interaction matrix.
      4. Build adjacency: edge between i and j iff `dist[i, j] < threshold`. Self-loops
         (diagonal entries are 0.0) are included but don't affect components.
      5. Find connected components -- the transitive closure of the near-clone relation.
      6. Return as pd.Series mapping bacteria name -> integer group label.

    Args:
        paper_repo: Path to the paper-repo clone (containing data/genomics/...).
        threshold: Core-genome distance threshold for grouping. Default 1e-4
            substitutions/site, matching the paper.

    Returns:
        pd.Series indexed by bacteria name with integer group labels in
        `[0, n_groups)`. Suitable for downstream use as in:
            from sklearn.model_selection import GroupKFold
            groups = load_strain_groups(paper_repo)
            gkf = GroupKFold(n_splits=10)
            for train_idx, test_idx in gkf.split(X, y, groups=groups.loc[X.index].values):
                ...
    """
    interaction_matrix = pd.read_csv(
        paper_repo / INTERACTION_MATRIX_RELPATH, sep=";"
    ).set_index("bacteria")
    distance_matrix = pd.read_csv(
        paper_repo / DISTANCE_MATRIX_ORIGNAMES_RELPATH, sep="\t"
    ).set_index("bacteria")

    strains = list(interaction_matrix.index)
    missing = [s for s in strains if s not in distance_matrix.index]
    assert not missing, (
        f"{len(missing)} interaction-matrix strain(s) missing from distance matrix: "
        f"{missing[:5]}"
    )

    # Subset to our 402 strains, preserving interaction-matrix order. The 2
    # "lost bacteria" in the distance matrix are dropped here implicitly.
    dist: np.ndarray = distance_matrix.loc[strains, strains].to_numpy()

    # Adjacency: edge iff distance strictly less than threshold. Diagonal is 0.0,
    # so self-loops are included but don't affect connected-component labeling.
    adjacency = (dist < threshold).astype(np.int8)
    _n_components, labels = connected_components(csr_matrix(adjacency), directed=False)

    return pd.Series(labels, index=pd.Index(strains, name="bacteria"), name="group")

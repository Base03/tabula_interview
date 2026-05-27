# Tabula Onsite Companion Document -- v2

A working reference for the Tabula bacterial-side prediction onsite, intended as the starting context for Claude Code (or another agent) attempting the work.

> **Notes for the agent reading this**
> 1. **Read this entire document before writing code.** It encodes decisions, corrections, and pitfalls that aren't obvious from the paper or the README alone.
> 2. **The first goal is to reproduce the paper, not to beat it.** See Sec.1.1 for why this is non-negotiable.
> 3. When in doubt, **read the actual training script** at `dev/predictions/predict_all_phages.py` (linked below) -- it differs from the README in important ways.
> 4. Tag conventions:
>    - `[VERIFIED-FROM-CODE]` -- extracted directly from the repo's source files (predict_all_phages.py).
>    - `[VERIFIED-BY-TRACE]` -- verified by running the trace scripts at `tests/test_paper_pipeline_structure.py` and `tests/test_strain_identifiers.py` against the published data.
>    - `[UNVERIFIED]` -- best estimate, not directly checked.

---

## 0. Implementation status and findings

Tracker for what's verified, what's coded, what's next. Tick boxes update as work progresses; findings are pinned one-liners so the doc stays a live record, not a wishlist.

### Phase 0 -- Setup `[done]`

- [x] **`paper-repo/`** cloned by `./scripts/bootstrap.sh`. Path-claim provenance verified against repo snapshot 2026-05-25 (see Sec.4.1).
- [x] **Trace scripts running.** `./scripts/test.sh` exercises 23 structural-contract assertions in `tests/test_paper_pipeline_structure.py` and `tests/test_strain_identifiers.py`. All green.

### Phase 1 -- Strain identifier mapping `[done]` `[VERIFIED-BY-TRACE]`

- [x] **402/403/404 row asymmetry pinned.** Interaction matrix has 402; Picard catalog has 403 (extra: `LF110`, has metadata but no interaction labels); distance matrix has 404 (extras: `H1-005-0065-L-P`, `H27` -- documented in `dev/predictions/visualize_predictions.ipynb` cell 2 as "lost bacteria", strains that had interaction data initially but were dropped before final analysis). The orphan sets do not overlap.
- [x] **Bacteria-name <-> distance-matrix bridge.** Use `370+host_distance_matrix_orignames.tsv` (indexed by friendly bacteria names directly). Picard's `Gembase` column is NOT the bridge -- different month code (`.0622.` vs `.0722.`) and only ~33/403 strains happen to overlap.
- [x] **Distance matrix well-formed.** Symmetric, diagonal is 0.0, self-distance equals row minimum (clone pairs tie at 0).
- [x] **3 perfect-clone pairs identified.** `ROAR047`/`ROAR072` and `H1-001-0020-M-O`/`H1-003-0090-V-J` are byte-identical row pairs (true duplicates). `IAI15`/`IAI17` is a near-duplicate: distance between them is 0 but their rows differ from each other by ~4e-6 -- independent PanACoTA runs of essentially identical genomes.

### Phase 2 -- Hand-crafted feature pipeline structure `[done]` `[VERIFIED-BY-TRACE]`

Already proven by `tests/test_paper_pipeline_structure.py`:

- [x] **18 pre-one-hot columns for normal phages** (not the 19 earlier doc drafts claimed). 13 for the 4 LF110 phages, which skip the phage-isolation-host merge and same_*_as_host engineering.
- [x] **114 post-one-hot dim for normal phages**, 109 for LF110 phages.
- [x] **H_host is phantom.** The line-57 regex filter drops `H-type` from `bact_features` before the line-79 rename runs, so `"H-type": "H_host"` is a silent no-op.
- [x] **`same_ABC_as_host` is a NaN-presence indicator, not constant True.** `NaN == NaN` is False in pandas, so the column collapses to "does this strain have an ABC capsule detected" rather than the always-True the doc originally claimed.
- [x] **Rare-binning at `< 3`** for O-type and ST_Warwick only (the inline `< 5` comment in the script is wrong; code uses 3).

### Phase 3 -- CV grouping `[done]` `[VERIFIED-BY-TRACE]`

- [x] **`src/cv.py`** -- `load_strain_groups(paper_repo)` returns sklearn-compatible group labels via connected components at $10^{-4}$ threshold on the orignames distance matrix.
- [x] **`tests/test_cv.py`** -- necessary and sufficient conditions, group count, clone-pair grouping, GroupKFold mechanics, plus pinning test for reconstruction-output stability (10 tests total).
- [x] **Reconstructed group statistics** (pinned in `test_cv.py`):
  - **301 groups** from 402 strains at the $10^{-4}$ threshold. Within the paper's claimed 250-350 range.
  - **244 singleton groups** (strains with no near-clones) + **57 non-singleton groups**.
  - **Largest group: 8 strains**. The 3 known clone pairs (Phase 1) all share a group ID with their partner, confirmed by `test_known_clone_pairs_share_a_group`.

### Phase 4 -- Hand-crafted featurizer implementation `[todo]`

- [ ] **`src/featurizers.py`** -- `Featurizer` ABC + `GaborieauHandCrafted` faithful to `predict_all_phages.py`, including the same_ABC_as_host self-compare bug intact for faithful reproduction.
- [ ] **`tests/test_featurizers.py`** -- regenerates the structural-contract pre/post-one-hot dims directly from our implementation.
- [ ] **Optional ablation switches** (toggle via constructor flag, default = faithful reproduction): fix same_ABC self-compare; add ABC_serotype to line-79 filter to get ABC_host; restore H-type to line-57 regex.

### Phase 5 -- Per-phage training and metrics `[todo]`

- [ ] **`src/training.py`** -- 4 model classes per phage (RF max_depth=3, RF max_depth=6, LogReg L2, LogReg L1), per-phage class-weight schedule (0.8 / 1 / 1.5 / 2 / 3), 10-fold GroupKFold using Phase-3 groups.
- [ ] **Per-phage best-of-4 selection by mean AUPR across test folds.**
- [ ] **`src/metrics.py`** -- AUROC, AUPR (paper's metrics) + within-host/phage Spearman/Pearson (brief's metrics), computed on the same out-of-fold probability outputs.
- [ ] **DummyClassifier baseline + per-phage-mean baseline** both included as floor references.

### Phase 6 -- Reproduction validation `[todo]`

- [ ] **Per-phage AUROC vs `paper-repo/dev/predictions/per_phage_perf.csv`** within tolerance. Specific fold-strain assignments will not match (we can't reproduce the missing `370+host_cross_validation_groups_1e-4.csv` exactly); per-phage AUROC distribution should match within ~0.05.
- [ ] **Aggregated AUROC** target: 0.86 (paper Fig 5B) and 0.77 per-phage average (paper Fig 5A).

### Phase 7 -- Bug ablations `[todo]`

Compare faithful reproduction (A) against fixes (B-D) on the same CV folds:

- [ ] **(A)** Faithful: same_ABC_as_host self-compare bug, no H_host, no ABC_host. Reference arm.
- [ ] **(B)** Fix same_ABC self-compare: add ABC_serotype to line-79 merge filter so ABC_host exists, then `same_ABC = ABC_serotype == ABC_host`. The "as intended" version.
- [ ] **(C)** Restore H-type to line-57 regex: H_host becomes real, joins to the engineered set as a "same H-type as phage's isolation host?" feature.
- [ ] **(D)** Both fixes simultaneously: (B) + (C).

Expected effect: small (paper says defense systems are marginal, surface features dominate). If (B)/(C)/(D) move AUROC by more than ~0.02, that's a finding worth flagging in any write-up.

---

## 1. Project context and approach

### 1.1 Reproduction-first principle

**Before any novel work, reproduce the paper's predictive model end-to-end on the supplied cross-validation folds and confirm your AUROC/AUPR per phage is in the same ballpark as `dev/predictions/per_phage_perf.csv`.**

This is the single most important methodological commitment in the document. There are five reasons:

1. **Calibration.** Without a reproduced baseline running through your evaluation harness, you have no calibrated number to compare any novel method against. Any "improvement" you measure could be a harness bug.

2. **Pipeline validation.** If reproduction is far off from the paper's published per-phage AUROCs, the cause is almost certainly a bug in *your* code (CV grouping, encoding, class weights, train/test split semantics) rather than a methodological insight. Reproduction is the only way to surface this.

3. **Trust signal for the interview.** Tabula will trust improvement claims more if you've first demonstrated you can faithfully reproduce existing work. Faithful reproduction signals intellectual honesty.

4. **De-risking.** Reproduction takes longer than expected (it always does). If reproduction consumes most of your 6 hours, you still have a deliverable -- "I reproduced the paper, here are the numbers" -- instead of a half-finished novel method with no baseline.

5. **It is the right comparison.** Any novel method must be compared against the paper's exact pipeline on the exact same folds. Reproducing the paper is the prerequisite for that comparison.

The realistic flow is **Reproduce -> Validate -> Extend**, not "skip reproduction and try cool things." Treat the time before novel extensions as cost, not as wasted effort.

### 1.2 Who, what, why

**Who:** A physics/ML person doing a 6-hour onsite at Tabula, a phage-therapy biotech developing treatments for antibiotic-resistant *Escherichia coli* infections. **No prior bacterial genomics background.**

**What:** Multi-output regression. Given a bacterial genome (FASTA assembly), predict a 96-dimensional vector of interaction scores against 96 fixed phages. Phages are *not* featurized -- they're fixed output dimensions. Score scale is ordinal `{0, 1, 2, 3, 4}` where higher = more potent lytic infection.

**Why this framing:** Tabula's deployment scenario is "patient presents with novel *E. coli* infection; given the bacterial genome, recommend phages from our fixed therapeutic library." That matches the "novel bacterium, known phages" task. Phage genomes exist publicly but are not in the project's task framing.

### 1.3 What Tabula is evaluating (in priority order)

1. **Reasoning** -- what you choose to try, why, how you evaluate it. They are in the room.
2. **Representations** -- the brief emphasizes "the main focus is the representation, not extensive tuning of the predictive head."
3. **Honesty** -- "an honest read on what seems to be working, what is not, and what we might try next."
4. **Comparison** -- against the no-model baseline and the hand-crafted reference, on the same folds. **This requires reproducing the paper first.**
5. **Clean code** -- the brief explicitly suggests a `featurizer.fit_transform(genomes)` API.
6. **Use of coding agents** -- "We suggest you use a coding agent extensively!"

What they are explicitly NOT asking for:
- A polished pipeline ("We do not expect a polished pipeline")
- A final answer / beating the paper
- Extensive hyperparameter tuning

---

## 2. The dataset

### 2.1 What's on disk

| Item | Description | Notes |
|---|---|---|
| 402 bacterial genomes | FASTA files of assembled contigs | ~5 Mbp each, 50-500 contigs per genome, ~2 GB total |
| Interaction matrix | Shape (402, 96), int values `{0..4}` | ~20% non-zero entries; 80% are 0 |
| Hand-crafted bacterial features | Likely the CSVs from the paper's GitHub | Confirm at onsite |
| Pre-computed genome representations | Internal to Tabula | **Ask what these are at the start.** Could include Mash sketches, ESM embeddings, Panaroo PA matrix |

**Phage genomes ARE in the public repo** at `data/genomics/phages/FNA/` (96 .fna files, one per phage) -- but the Tabula brief's task framing treats phages as fixed output dimensions only, so they're not featurized. Confirm at the onsite whether the brief is providing them and whether featurizing phages is in scope.

### 2.2 Crucial scale facts

- **Effective sample size: 402** (or ~250-350 after grouping near-clones for CV).
- Output dimensionality: 96 (correlated -- phages from same viral genus have correlated host ranges).
- Total labeled scalars: $402 \times 96 = 38{,}592$ in the published interaction matrix (paper text says $403 \times 96 = 38{,}688$; see Sec.2.4). Either way, only 402 distinct inputs.
- Raw genome size: ~5,000,000 ACGT characters per strain. Almost all of this is irrelevant.
- This is **UCI-tabular-dataset scale**, not deep-learning scale.

### 2.3 Score semantics

The 0-4 score, called **MLC** (Minimum Lytic Concentration), is the lowest phage concentration at which a lytic interaction was observed in plaque assay. From the paper Methods (and `data/interactions/raw/raw_interactions.csv`), replicate coverage is **asymmetric across MOI**:

| MOI | Phage titer (PFU/mL) | Replicates |
|---|---|---|
| 10 | $5 \times 10^8$ | R1, R2, R3 |
| 1 | $5 \times 10^7$ | R2, R3 |
| 0.1 | $5 \times 10^6$ | R1, R2, R3 |
| 0.001 | $5 \times 10^4$ | R1 only |

MLC scoring uses only MOI {10, 1, 0.1}; MOI 0.001 results are excluded because they're unreplicated. The score is:

- **0**: no lytic interaction at any MOI.
- **1**: lysis at MOI 10 only (highest titer).
- **2**: lysis at MOI 1 (mid titer).
- **3**: individual lysis plaques at MOI 0.1 (lowest counted titer).
- **4**: uncountable plaques / full lawn clearing at MOI 0.1.

Paper reports 98.35% of (phage, bacterium, MOI) triples yield consistent scores across triplicates (same score in all 3 reps in 88.64%; same in 2/3 in 9.71%).

**Note on the ColoColi cocktail-validation experiment** (separate from the main matrix): six MOI levels {10, 1, 0.1, 0.01, 0.001, 0.0001} are used, so MLC scores there range 0-6.

The paper **binarized** the 0-4 scores to lytic vs non-lytic for its AUROC = 0.86 aggregated / 0.77 per-phage-average headline numbers. The Tabula brief asks for **ordinal regression on 0-4**.

### 2.4 The 402-vs-403 anomaly

The paper text consistently reports **403 strains x 96 phages = 38,688 interactions**, and the strain catalog `picard_collection.csv` in the repo has 403 rows. But the published `interaction_matrix.csv` has only **402 rows** -- one catalog strain has metadata but no interaction labels. Tabula's brief ships 402 rows (matching the CSV), so the **working dataset is 402 strains**.

When this doc cites a number, it follows this convention:
- "**402**" when describing the working data (what you actually train on, what's in `interaction_matrix.csv`, what Tabula delivers).
- "**403**" when citing the paper's catalog headline or the strain metadata CSV (e.g., quoted paper findings about the Picard collection).

The missing strain is not a designated holdout, just an asymmetry in the published artifacts. The cause is unclear (possibly a failed assay row dropped before publication, possibly an off-by-one in the paper headline).

---

## 3. The paper

### 3.1 Citation

Gaborieau, B., Vaysset, H., Tesson, F., et al. *Prediction of strain level phage-host interactions across the Escherichia genus using only genomic information.* **Nature Microbiology** 9:2847-2861 (2024). DOI: [10.1038/s41564-024-01832-5](https://doi.org/10.1038/s41564-024-01832-5)

- **PubMed**: [39482383](https://pubmed.ncbi.nlm.nih.gov/39482383/)
- **bioRxiv preprint**: [2023.11.22.567924](https://www.biorxiv.org/content/10.1101/2023.11.22.567924v1)
- **HAL accepted manuscript** (open access): [hal-04781608](https://hal.science/hal-04781608v1/file/Gaborieau_et_al_Accepted%20Manuscript.pdf)

### 3.2 Headline result

Binarized lytic vs non-lytic, ten-fold group cross-validation across the 403 strains in the paper's experimental matrix (working dataset is 402; see Sec.2.4). Paper reports **two** AUROC numbers from the same CV: **0.77 averaged across the 96 per-phage classifiers** (Fig 5A) and **0.86 on the aggregated prediction matrix** (Fig 5B). The aggregated number is what gets quoted in the abstract; the per-phage average is the more honest single-classifier metric. Per-phage performance varies; some narrow-host-range phages ($<5$ lytic interactions out of 402) are poorly predicted by any method.

**Internal paper inconsistency to note:** the Discussion (p.22) reports the aggregated number as "**85%**" while Results / Fig 5B (p.17) says **86%**. The Results number is more authoritative; the Discussion likely rounded down. If reviewers cite "85% per the paper," they're quoting the Discussion.

### 3.3 Key biological conclusion

**"Bacterial adsorption factors are the major determinants of phage-bacteria interactions in contrast with defence systems which marginally reduce virulence of infecting phages."** (Figure 4 title.)

Spend representation budget on surface features. Defense systems are not high-EV.

### 3.4 Selected paper-specific numbers worth knowing

- **Picard collection composition (n=403 per paper; published interaction matrix has 402):** 378 *E. coli* + 25 other *Escherichia* species (*E. fergusonii*, *E. albertii*, *E. ruysiae*, *E. marmotae*, plus Clades I-V). 163 STs, 93 O-antigen types, 41 H-antigen types, 5 outer-core types (R1, R2, R3, R4, K12; n=36 untypeable).
- **Capsules:** 22 strains (5.4%) carry a *Klebsiella*-style group-1 capsule (all in phylogroups A/B1/C with O-types O8/O9/O89). 171 strains carry an ABC-dependent (group-2/3) capsule; 101 of those were K-typed. K-antigens identified: K1 (n=42), K5 (n=19), K4 (n=11), K2 (n=10), K15 (n=6), K7 (n=6), K10 (n=1).
- **Phage receptor OMPs:** 12 OMPs tracked, consistent between Fig 1H and the repo. TSV column order in `data/genomics/bacteria/outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv`: `BTUB, FADL, FHUA, LAMB, LPTD, NFRA, OMPA, OMPC, OMPF, TOLC, TSX, YNCD`. Each found in >97% of strains. Worth knowing if you go looking: **FepA** -- a canonical literature *E. coli* phage receptor -- is NOT in Gaborieau et al.'s tracked set; NfrA (phage N4 receptor) is included instead.
- **Defense systems:** 137 distinct subfamilies catalogued in the *Escherichia* pan-immune system; mean 8 per isolate (range 1-16).
- **The K1-capsule shared-RBP story (Fig 3C-D)** is the paper's clearest mechanistic finding: three phages from two different genera (T145_P2 and AL505_P3 are *Vectrevirus*; MT1B1_P3 is *Kayfunavirus*) share a $\beta$-propeller / $\beta$-barrel / tailspike RBP architecture homologous to phage K1F's K1-targeting tailspike (PDB 3GVJ). Across the *Picard* panel, the rule "lyse iff strain encodes K1 capsule" achieves **73% precision, 72% recall** for these three phages -- compared to 7% precision / 14% recall for the same rule applied to all other *Vectrevirus* / *Kayfunavirus* phages. Concrete evidence that RBP identity (not isolation host) drives host range.
- **Cocktail recommender validation (Fig 6):** evaluated on 100 *E. coli* VAP strains from the ColoColi collection (310 strains total, 100 selected for test). 18 distinct cocktail compositions recommended across 100 strains; "tailored" cocktails were assigned at early pipeline stages (15 cocktails for 24 strains), "generic" fallbacks at late stages (3 cocktails for 76 strains). Lytic success rates on test strains: **tailored 91.67%, baseline 81.00%, generic 78.95%**. Tailored significantly outperforms both (Mann-Whitney p=0.02). **Cocktail size:** the paper picked k=3 phages because the *most-covering triplet* of phages lyses 63% of the *Picard* matrix and the *most-covering 4-uplet* lyses 68% (Supplementary Figure 7) -- the small marginal gain didn't justify the larger cocktail. These coverage numbers are over the training matrix, not the held-out test set.

---

## 4. Resource locations

### 4.1 Primary repository

**GitHub: [`mdmparis/coli_phage_interactions_2023`](https://github.com/mdmparis/coli_phage_interactions_2023)**

Path inventory below was verified by a fresh `git clone` on 2026-05-25. The repo isn't versioned with stable tags, so sanity-check with `find data dev -maxdepth 3 -type d` on first clone in case anything moved.

Key paths:
- [`README.md`](https://github.com/mdmparis/coli_phage_interactions_2023/blob/main/README.md) -- read first, but cross-check against the script.
- [`dev/predictions/predict_all_phages.py`](https://github.com/mdmparis/coli_phage_interactions_2023/blob/main/dev/predictions/predict_all_phages.py) -- **the authoritative training script. Diverges from the README in non-trivial ways.**
- [`data/genomics/bacteria/picard_collection.csv`](https://github.com/mdmparis/coli_phage_interactions_2023/blob/main/data/genomics/bacteria/picard_collection.csv) -- feature table per strain (semicolon-separated).
- [`data/genomics/bacteria/umap_phylogeny/coli_umap_8_dims.tsv`](https://github.com/mdmparis/coli_phage_interactions_2023/blob/main/data/genomics/bacteria/umap_phylogeny/coli_umap_8_dims.tsv) -- pre-computed UMAP coordinates.
- [`data/interactions/interaction_matrix.csv`](https://github.com/mdmparis/coli_phage_interactions_2023/blob/main/data/interactions/interaction_matrix.csv) -- the ($402 \times 96$) score matrix (paper text says $403 \times 96$; see Sec.2.4).
- `data/interactions/raw/raw_interactions.csv` -- raw plate-by-plate annotations before aggregation.
- `data/genomics/bacteria/outer_membrane_proteins/` -- a single TSV `blast_results_cured_clusters=99_wide.tsv` of cluster IDs per strain x OMP. **NOT pre-extracted sequences** -- to get amino-acid sequences you still need to annotate with Bakta/Prokka and join via locus tags.
- `data/genomics/bacteria/defense_finder/` -- shipped DefenseFinder outputs: `370+host_defense_systems_subtypes.csv`, `370+host_defense_systems_cluster_80_80.csv`, `defense_arsenal_370+host.pickle`. Skip running DefenseFinder yourself.
- `data/genomics/bacteria/panacota/tree/` -- the phylogenetic tree (Newick).
- `data/genomics/bacteria/isolation_strains/panacota/tree/370+host_distance_matrix.tsv` -- **pre-computed pairwise phylogenetic distance matrix.** Use this directly for CV grouping; no need to parse the Newick yourself.
- `data/genomics/phages/FNA/` -- 96 phage genome FASTAs (yes, phage genomes are shipped).
- `dev/predictions/per_phage_perf.csv` -- per-phage AUROC/AUPR for the *aggregated/best* model. Only one model column ("AF"); the per-phage best-of-4 selection details and full per-phage CSVs live in `dev/predictions/results/performances/performance_<phage>_Group10Fold_CV.csv` (96 files).
- `dev/predictions/results/feature_importances.rar` -- per-phage feature importances.
- `dev/predictions/results/predictions.rar` -- per-phage prediction outputs (binarized).
- `dev/inference_bacteria/phage_per_phage/` -- LMM analyses for Figure 4.
- `dev/cocktails/` -- Figure 6 phage cocktail recommendation analysis.

**Critical missing file:** `370+host_cross_validation_groups_1e-4.csv` -- referenced in the training script (path `D:\\These\\30_dev\\...` on the author's local drive) but not in the repo. README explicitly notes it's "not available anymore." Recompute from the shipped pairwise distance matrix or via `skani` ANI as a proxy.

### 4.2 Data DOIs

- **Genome assemblies (FASTA)**: figshare [10.6084/m9.figshare.25941691.v1](https://doi.org/10.6084/m9.figshare.25941691.v1)
- **Raw plaque assay images & matrix**: Zenodo [10.5281/zenodo.10202713](https://doi.org/10.5281/zenodo.10202713)
- **Code snapshot**: Zenodo [10.5281/zenodo.13831957](https://doi.org/10.5281/zenodo.13831957)
- **Interactive viewer**: [viralhostrangedb.pasteur.cloud](https://viralhostrangedb.pasteur.cloud/)

### 4.3 Bioinformatics tools

| Tool | Purpose | Repo |
|---|---|---|
| Prodigal | Gene calling | [hyattpd/Prodigal](https://github.com/hyattpd/Prodigal) |
| pyrodigal | Python bindings | [althonos/pyrodigal](https://github.com/althonos/pyrodigal) |
| Prokka | Annotation wrapper (faster) | [tseemann/prokka](https://github.com/tseemann/prokka) |
| Bakta | Annotation (more thorough) | [oschwengers/bakta](https://github.com/oschwengers/bakta) |
| PanACoTA | Pan-genome + phylogeny (used in paper) | [gem-pasteur/PanACoTA](https://github.com/gem-pasteur/PanACoTA) |
| MacSyFinder | Multi-gene system detection (used in paper) | [gem-pasteur/macsyfinder](https://github.com/gem-pasteur/macsyfinder) |
| ECTyper | O/H serotyping (used in paper) | [phac-nml/ecoli_serotyping](https://github.com/phac-nml/ecoli_serotyping) |
| Kaptive | Capsule serotyping (used in paper) | [katholt/Kaptive](https://github.com/katholt/Kaptive) |
| ClermonTyping | Phylogroup typing | [A-BN/ClermonTyping](https://github.com/A-BN/ClermonTyping) |
| SRST2 | MLST sequence typing | [katholt/srst2](https://github.com/katholt/srst2) |
| DefenseFinder | Antiphage defense systems | [mdmparis/defense-finder](https://github.com/mdmparis/defense-finder) |
| skani | Fast ANI (for CV groups) | [bluenote-1577/skani](https://github.com/bluenote-1577/skani) |
| Mash / sourmash | k-mer sketches | [marbl/Mash](https://github.com/marbl/Mash), [sourmash-bio/sourmash](https://github.com/sourmash-bio/sourmash) |

### 4.4 Learned-representation tools

| Tool | Purpose | Location |
|---|---|---|
| ESM-2 (8M, 35M, 150M, 650M, 3B, 15B) | Protein language model | [facebookresearch/esm](https://github.com/facebookresearch/esm), HF `facebook/esm2_t12_35M_UR50D` |
| gLM2 | Mixed-modality genomic LM | HF [`tattabio/gLM2_650M`](https://huggingface.co/tattabio/gLM2_650M) |
| Bacformer | Genome-context protein LM | HF [`macwiatrak/bacformer-masked-MAG`](https://huggingface.co/macwiatrak/bacformer-masked-MAG) |

---

## 5. Biology primer (minimum required)

### 5.1 The DNA -> protein -> surface pipeline

1. A **bacterial genome** is one big ACGT string (~5 Mb in *E. coli*), one circular chromosome. After short-read assembly: 50-500 linear contigs summing to ~5 Mb.
2. **Genes** are ~5,000 substrings of length 300-3000, each encoding one protein. **Prodigal** finds them by ATG-to-stop-codon stretches scored by codon-usage statistics.
3. **DNA->protein translation is a fixed deterministic lookup** (the genetic code; 3 DNA letters -> 1 amino acid).
4. **Proteins do work** in the cell. Some sit on the outer surface.
5. **Sugars on the surface are built by enzymes** which are encoded by DNA. So "this strain has O-antigen type X" is determined by "this strain has enzyme cluster X."

### 5.2 What's on the *E. coli* surface

- **LPS** (Lipopolysaccharide):
  - **Lipid A** (anchor, conserved)
  - **Core oligosaccharide** (~5 outer-core types: R1, R2, R3, R4, K-12)
  - **O-antigen**: long polysaccharide, >180 known variants. Built by the ***rfb*** cluster. Diagnostic genes: ***wzx***, ***wzy***.
- **Capsule** (optional extra polysaccharide layer):
  - **Group 1 / Klebsiella-style**: ***cps/wba*** cluster (Kaptive detects).
  - **Group 2/3 / ABC-dependent**: ***kps*** cluster (CapsuleFinder/MacSyFinder detects).
- **OMPs** (Outer Membrane Proteins): $\beta$-barrels in the outer membrane.
  - **LamB** -- maltose porin, receptor for phage $\lambda$
  - **OmpA, OmpC, OmpF** -- porins
  - **FhuA** -- ferrichrome transporter (T1, T5, $\phi$80)
  - **BtuB** -- vitamin B12 receptor
  - **FepA** -- enterobactin receptor; canonical literature OMP for *E. coli* phages but Gaborieau et al. did not include it in their tracked set (Fig 1H and the TSV both use NfrA in this slot).
  - **NfrA** -- outer-membrane receptor for phage N4; one of the 12 OMPs in Fig 1H and the clustering TSV.
  - **LptD** -- LPS export
  - **Tsx** -- nucleoside channel
  - **TolC** -- efflux outer membrane channel
- **Pili, flagella** -- appendages, sometimes used as receptors.

### 5.3 The infection cascade

1. **Adsorption** -- phage docks to surface receptor. **Dominant specificity step.**
2. **DNA injection** -- phage punctures membrane.
3. **Intracellular replication** -- defense systems may abort (R-M, CRISPR, abortive infection, CBASS, BREX, retrons, etc.). The paper finds these matter "marginally" vs adsorption.
4. **Lysis** -- cell breaks open.

### 5.4 Phylogeny terminology

- **Phylogroup**: 8 coarse classes (A, B1, B2, C, D, E, F, G) for *E. coli*.
- **MLST / Sequence Type (ST)**: finer classification based on 7 housekeeping genes. 163 distinct STs in the Picard collection.
- **Core genome**: ~3,500 genes present in $\geq 99\%$ of strains.
- **Accessory / pan genome**: ~15,000-30,000 genes total across all strains.
- **ANI**: Average Nucleotide Identity between two genomes. ANI $\geq 99.99\%$ = effectively clonal.

---

## 6. The actual hand-crafted feature pipeline `[VERIFIED-FROM-CODE]`

This section is derived directly from `dev/predictions/predict_all_phages.py` and the `picard_collection.csv` data file. **The training script diverges from the README in important ways** -- when they conflict, trust the script.

### 6.1 Feature filter regex

The script applies this regex to the merged bacterial features + UMAP table:

```python
bact_feat_names = "(UMAP|O-type|LPS|ST_Warwick|Klebs|ABC_serotype)"
bact_features = bact_features.filter(regex=bact_feat_names, axis=1)
```

This selects:

| Pattern | Columns matched | Type | Count |
|---|---|---|---|
| `UMAP` | UMAP_1...UMAP_8 | continuous | 8 |
| `O-type` | O-type | categorical | 1 |
| `LPS` | LPS_type | categorical | 1 |
| `ST_Warwick` | ST_Warwick | categorical | 1 |
| `Klebs` | Klebs_capsule_type | categorical | 1 |
| `ABC_serotype` | ABC_serotype | categorical | 1 |

**Total: 13 input columns from the bacterial side.**

### 6.2 Engineered features (NOT in the README)

The script also merges in features of the phage's isolation strain and creates "same as host" booleans:

```python
# Add phage-isolation-host features
phage_host_features = pd.merge(phage_feat, bact_features.filter(
    regex="(ST_Warwick|O-type|H-type)", axis=1
), left_on="Phage_host", right_index=True).rename(
    # Full rename dict in the script also includes Clermont_Phylo -> Clermont_host,
    # LPS_type -> LPS_host, and H-type -> H_host -- but those columns are filtered
    # out before this merge (none of them survive line 57's regex), so the renames
    # are silent no-ops. Result: phage_host_features ends up with only ST_host and
    # O_host (plus Phage_host which is dropped on line 82).
    {"O-type": "O_host", "H-type": "H_host", "ST_Warwick": "ST_host"}, axis=1
)

# Engineered booleans
interaction_with_features["same_O_as_host"]      = ...["O-type"] == ...["O_host"]
interaction_with_features["same_ST_as_host"]     = ...["ST_Warwick"] == ...["ST_host"]
interaction_with_features["same_ABC_as_host"]    = ...["ABC_serotype"] == ...["ABC_serotype"]  # WARNING: BUG
interaction_with_features["same_O_and_ST_as_host"] = same_O x same_ST
# O_host is dropped after creating same_O_as_host; H_host never existed.
```

**Pre-one-hot total: 18 columns** (verified empirically across all 92 non-LF110 phages by running the trace at `dev/verify_predict_pipeline.py`):

| | Count |
|---|---|
| UMAP_1...8 (continuous) | 8 |
| O-type, LPS_type, ST_Warwick, Klebs_capsule_type, ABC_serotype (bacteria, categorical) | 5 |
| ST_host (only phage-isolation-host feature; H_host is a phantom rename target -- doesn't exist post-line-57 filter) | 1 |
| same_O_as_host, same_ST_as_host, same_ABC_as_host, same_O_and_ST_as_host (boolean) | 4 |
| **Total** | **18** |

**LF110 carve-out (4 phages, `LF110_P1`..`LF110_P4`):** the script skips both the phage-host merge (line 82, `if not p.startswith("LF110")`) and all four `same_*_as_host` boolean creations. Those 4 phages run with **13 pre-one-hot columns** (just the 8 UMAPs + 5 bacterial categoricals) and **109 post-one-hot**. The comment in the script says "do not have the data for LF110 host strain". Tracker doc, don't forget these when reproducing.

### 6.3 The same_ABC_as_host bug (subtler than it looks)

```python
interaction_with_features["same_ABC_as_host"] = interaction_with_features["ABC_serotype"] == interaction_with_features["ABC_serotype"]
```

Two things are wrong here, and the second isn't obvious:

1. **The self-compare.** This was almost certainly a copy-paste error from `same_O_as_host` -- the engineer presumably meant `["ABC_serotype"] == ["ABC_host"]`. But ABC_host doesn't exist either: the regex on line 79 only pulls `(ST_Warwick|O-type|H-type)` from bact_features, so an ABC_host column never gets created. Fixing line 101 alone wouldn't fix the feature; you'd also need to add `ABC_serotype` to the line 79 merge filter.

2. **It's NOT always True.** In pandas, `NaN == NaN` evaluates to **False**. Many strains have no ABC capsule and so `ABC_serotype` is NaN. Empirically, `same_ABC_as_host` is True for strains with a detected ABC serotype and False for strains without. So this "bug" effectively becomes an **"ABC capsule was detected" presence indicator** -- not a constant column, and not what the engineer intended, but it carries real signal that the model can and does use after one-hot encoding.

**When reproducing, replicate this exactly** for faithful reproduction. Fixing it to compare against an actual ABC_host (adding `ABC_serotype` to the line 79 filter) is a clean experiment to try as an extension, but separate from reproduction itself.

### 6.4 Rare-category binning

Applied only to O-type and ST_Warwick (NOT to ABC_serotype, Klebs_capsule_type, LPS_type, ST_host):

```python
otypes_to_recode = bact_features.groupby("O-type").filter(lambda x: x.shape[0] < 3)["O-type"].unique()
interaction_with_features.loc[interaction_with_features["O-type"].isin(otypes_to_recode), "O-type"] = "Other"
# Same logic for ST_Warwick
```

So categories with **fewer than 3 occurrences** get binned into "Other" -- but only for two of the six categorical features. **Heads-up:** the inline code comments at `predict_all_phages.py:86` and `:94` say "less than 5 observations" -- the comments are wrong; the code uses `< 3`. Trust the code.

### 6.5 One-hot expansion and post-encoding dimensionality

```python
X_oh = pd.concat([
    (X[num] - X[num].mean(axis=0)) / X[num].std(axis=0),  # standardize 8 UMAPs
    pd.get_dummies(X[factors], sparse=False)              # one-hot the rest
], axis=1)
```

Measured category counts after one-hot (empirical, from `dev/verify_predict_pipeline.py`):

| Column | Distinct values | Post-binning expansion |
|---|---|---|
| LPS_type | R1, R2, R3, R4, K12, No_waaL | ~6 (no binning needed) |
| O-type | ~80-90 in panel; many singletons | ~25-35 + "Other" |
| ST_Warwick | 163 in panel; many singletons | ~25-35 + "Other" |
| Klebs_capsule_type | mostly blank; visible: K2, K9, K10, K16, K25, K39, K54, K55, K57, K63 | ~11 (no binning) |
| ABC_serotype | blank, HP, Unknown, CatB, K1, K2, K4, K5, K7, K10, K15, K98, 1, 3, 5 | ~14 (no binning) |
| ST_host | only STs of the 34 phage isolation strains; varies per phage | ~15-25 |
| same_O/ST/O+ST_as_host (3 booleans, 2 unique values each) | 2 dummy cols each | 6 |
| same_ABC_as_host (1 dummy: True/False maps to "has ABC capsule" / "doesn't") | 1 dummy col | 1 |
| UMAP | 8 continuous | 8 |
| **Post-one-hot total (measured: 114 for 92 normal phages, 109 for the 4 LF110 phages)** | | **109-114 dims** |

So the **input to the model is 109-114 dims for the published data**, not the "~120-150" earlier draft estimate. The 18-column count is pre-one-hot. The drop from prior estimates: H_host (~10-15 dummies) doesn't exist, and `same_ABC_as_host` only contributes 1 dummy (not 2) because it has one unique value per the NaN-handling described in Sec.6.3.

### 6.6 Per-column gene-level breakdown

What each column actually probes in the genome:

| Feature | Underlying genes | Information type | Notes |
|---|---|---|---|
| **UMAP_1...8** | ~3,500 core genes (housekeeping) | Phylogeny (continuous) | Abstract coordinates from UMAP-reduced core-genome distance |
| **O-type** | *wzx*, *wzy* (diagnostic); biology lives in full *rfb* cluster of 10-25 genes | Real surface receptor (categorical proxy) | Cluster genes vary in composition between O-types |
| **LPS_type** | *waaL* (diagnostic, single gene); biology in *waa* operon (~10 genes) | Real surface receptor | Only 6 categories, all well-populated |
| **ST_Warwick** | 7 housekeeping genes (*adk*, *fumC*, *gyrB*, *icd*, *mdh*, *purA*, *recA*) | Phylogeny (discrete) | **Largely redundant with UMAP -- both encode phylogenetic position** |
| **Klebs_capsule_type** | *cps/wba* cluster (~15-25 genes when present) | Real surface receptor (when present) | Most strains are blank |
| **ABC_serotype** | *kps* cluster (~6-10 genes when present) | Real surface receptor (when present) | Has data hygiene issues -- `1`/`3`/`5` likely duplicate K1/K3/K5 |
| **ST_host** | Same 7 genes for phage's isolation strain | Phylogeny of phage source | Side channel about which strain phage came from |
| **same_*_as_host (booleans)** | Derived from above | Engineered interaction terms | Capture "test strain matches phage's known host"; `same_ABC_as_host` is the self-compare bug (see Sec.6.3) |

### 6.7 Phylogenetic redundancy is substantial

| Feature group | Genes | Post-one-hot dim | Information type |
|---|---|---|---|
| UMAP_1...8 | ~3,500 core genes | 8 | Phylogeny (continuous) |
| ST_Warwick | 7 genes | ~25-35 | Phylogeny (discrete) |
| ST_host | Same 7 genes | ~15-25 | Phylogeny of source |
| **Phylogeny subtotal** | | **~50-70** | All measuring related things |
| O-type | *rfb* | ~30 | Real receptor |
| LPS_type | *waa* | 6 | Real receptor |
| Klebs_capsule_type | *cps/wba* | ~11 | Real receptor |
| ABC_serotype | *kps* | ~14 | Real receptor |
| **Receptor subtotal** | | **~60** | Surface mechanism |
| same_*_as_host | derived | 7 (3 booleans x 2 + same_ABC x 1) | Engineered |

**Roughly half the feature vector is phylogeny encoded three different ways** (UMAP, ST_Warwick, ST_host). The other half is what actually probes surface receptors, and within that half most strains light up only a few columns.

### 6.8 Model training pipeline

```python
# Note: paper Methods text describes 4 model classes; DummyClassifier is in the
# code but not the paper text. Treat it as a code-only stratified-random baseline.
models_to_test = [
    RandomForestClassifier,
    RandomForestClassifier,
    LogisticRegression,
    LogisticRegression,
    DummyClassifier  # the baseline
]

# Class weights chosen by per-phage positive prevalence
perc_pos_class = y.sum() / y.shape[0]
if 0.60 <= perc_pos_class:               cw = {0: 1, 1: 0.8}
elif 0.4 <= perc_pos_class < 0.6:        cw = {0: 1, 1: 1}
elif 0.3 <= perc_pos_class < 0.4:        cw = {0: 1, 1: 1.5}
elif 0.2 <= perc_pos_class < 0.3:        cw = {0: 1, 1: 2}
else:                                    cw = {0: 1, 1: 3}

params = [
    {"max_depth": 3, "n_estimators": 250, "class_weight": cw},
    {"max_depth": 6, "n_estimators": 250, "class_weight": cw},
    {"class_weight": cw, "max_iter": 10000},
    {"class_weight": cw, "penalty": "l1", "solver": "saga", "max_iter": 10000},
    {"strategy": "stratified"}
]
```

- **GroupKFold with 10 splits**, groups defined by core-genome distance $<10^{-4}$.
- Per-phage best model selected by mean AUPR across folds.
- `sklearn` v1.1.2.

#### How class_weight actually behaves

`class_weight={0: 1, 1: w}` is a **per-class** multiplier, not a per-sample uniform scaling. The training loss becomes:

$$\mathcal{L}_\text{weighted} = \sum_{i:\, y_i = 0} \ell_i + w \cdot \sum_{i:\, y_i = 1} \ell_i$$

Class-0 samples count once, class-1 samples count $w$ times. The two classes are not multiplied uniformly, so this tilts the loss landscape between them rather than rescaling it overall.

- **For LogisticRegression** (with L2 default): the optimum $\theta$ shifts because positive samples exert $w \times$ the gradient pressure during fitting, while the L2 regularization term is unchanged. Effectively, raising $w$ reduces the regularization's relative pull on features that discriminate positives.
- **For RandomForestClassifier**: the Gini/entropy criterion at each split substitutes weighted counts for raw counts, so the tree structure itself changes -- splits that purify positives are weighted more heavily.

**Even though we train one model per phage**, the per-class weighting still does work. Without it, a 5%-positive phage's classifier would be trained on a much more imbalanced loss surface than a 50%-positive phage's, and the 96 classifiers would converge under qualitatively different training dynamics. The schedule normalizes that within-phage class imbalance regime across phages, so the trained classifiers are comparable.

Why a step function instead of sklearn's `class_weight="balanced"` (which sets $w = N_\text{neg} / N_\text{pos}$ automatically)? For a 5%-positive phage, "balanced" gives $w \approx 19$, which destabilizes LogReg and inflates RF feature importances at the rare class. Capping at 3 is a heuristic ceiling that keeps training stable.

#### Out-of-fold prediction structure

Across the 10 folds, every strain appears in exactly **one** test fold. So each strain has exactly one prediction, made by the model trained on the 9 folds that didn't include it. The paper's headline AUROC is computed on the concatenation of those 10 fold-level test predictions -- 402 strain-level predictions per phage, from 10 different trained models.

**The paper never trains a "final" model on all 402 strains.** Each phage gets 50 saved models (5 model classes x 10 folds) at training time; per-phage best-of-5 selection picks one model class by mean AUPR, leaving 10 fold-models for that class. For a new bacterium (not in the 402), the paper has options none of which are perfect: pick one fold-model arbitrarily, ensemble all 10 by averaging probabilities, or retrain on all 402. The cocktail-recommender code (`dev/cocktails/` in the paper repo) is where you'd see what they actually did for the 100-strain ColoColi test set in Fig 6; not traced here yet.

---

## 7. Evaluation

### 7.1 The four metric variants (per the Tabula brief)

For each test fold, predict `Y_pred` (n_test, 96), compare to `Y_true`.

**Within-host correlation** (rank phages correctly per strain?):
```python
within_host = mean(corr(Y_pred[i, :], Y_true[i, :]) for i in range(n_test))
```

**Within-phage correlation** (rank strains correctly per phage?):
```python
within_phage = mean(corr(Y_pred[:, j], Y_true[:, j]) for j in range(96))
```

Both with Spearman and Pearson, averaged across folds. **Report all four numbers with stdev.**

### 7.2 The asymmetric baseline behavior

The no-model baseline `Y_pred[i, :] = Y_train.mean(axis=0)`:

- **Within-host Pearson is substantial** (~0.3-0.5) because the matrix has shared structure.
- **Within-phage Pearson is ~0** because prediction is constant across test strains for each phage.

**Within-phage Pearson is the honest test of whether features carry signal.**

### 7.3 Phylogenetic data leakage and CV grouping

Strains within core-genome distance $<10^{-4}$ substitutions/site must be in the same fold. The original grouping file isn't in the repo; reconstruct via:

**Option A (use the shipped matrix):** the repo ships `data/genomics/bacteria/isolation_strains/panacota/tree/370+host_distance_matrix_orignames.tsv`, a 404 x 404 pre-computed pairwise phylogenetic distance matrix indexed by friendly bacteria names (the `_orignames` variant -- the non-orignames file uses sequential PanACoTA build IDs `ESCO.0722.NNNNN` which DON'T match picard's `Gembase` column; see `test_strain_identifiers.py` for the structural-contract test pinning the naming). Threshold at $10^{-4}$ substitutions/site, find connected components.

**Option B (parse Newick yourself):** load the Newick tree at `data/genomics/bacteria/panacota/tree/370+host_ultrametric_tree_root=B992.nwk` (the paper's LMM code uses this -- leaves are bacteria-named directly, `bact_phylo_tree$tip.label` per `dev/inference_bacteria/phage_per_phage/phage_per_phage_MCMC_family=binom.Rmd:79-87`). Compute cophenetic pairwise distances. Threshold at $10^{-4}$, find connected components. `[UNVERIFIED]` Should be numerically equivalent to Option A -- both ultimately derive from PanACoTA's output of the same tree -- but we haven't checked this directly (would require a Newick parser dependency we don't otherwise need). If you go this route, sanity-check against Option A on a few strain pairs first. Worth verifying in a larger project, out of scope here.

**Option C (fast proxy):** Run `skani triangle` on the 402 FASTAs, threshold at ANI $\geq 99.99\%$, find connected components. ~1 minute on a laptop. Doesn't require the distance matrix or Newick.

#### Grouping is by connected component, not pairwise

The threshold rule "strains within $d < 10^{-4}$ are in the same fold" gets one detail wrong if read literally: **same-fold-ness is by connected component of the thresholded graph, not by pairwise distance.** Three strains A, B, C where $d(A,B) < 10^{-4}$ and $d(B,C) < 10^{-4}$ but $d(A,C) > 10^{-4}$ are all in the same group via B. Transitive closure.

Why this matters:
- A clade with strains spaced $0.5 \times 10^{-4}$ apart can collapse into one large group whose total diameter is well above $10^{-4}$, far larger than naive pairwise inspection suggests.
- The reverse misreading -- "if two strains are >$10^{-4}$ apart they can be in different folds" -- is wrong. They might still need to share a fold via intermediates.

When asserting fold correctness in tests, distinguish:

- **Necessary** (cheap to check): for every pair $(i, j)$ with $d_{ij} < 10^{-4}$ directly, group$(i) =$ group$(j)$.
- **Sufficient** (the actual paper rule): group assignments are the connected components of the graph thresholded at $10^{-4}$ (transitive closure).

Both should be tested. Failing the necessary condition means the grouping is broken. Passing necessary but failing sufficient means transitively-near strains got split across folds, which is the subtle data-leakage bug the paper's threshold is designed to prevent.

#### Expectations for fold-level reproduction

`sklearn.model_selection.GroupKFold` has **no `random_state`** parameter -- it's deterministic given (a) the group-label integers and (b) the order of samples passed in. Its algorithm sorts groups by size descending and greedily assigns each to the currently-smallest fold, breaking ties by group ID. So fold-level reproduction depends on whether our group labels and sample ordering match the paper's, which we can't verify (the paper's grouping code is gone). **Expect different specific fold assignments**; don't write tests that assert exact fold membership.

What to assert instead: the *invariants* that any correct grouping must satisfy (the necessary and sufficient conditions above), and that the *number* of effective groups falls in the expected range (paper claims ~250-350 effective independent strains after grouping).

### 7.4 Paper-numbers caveat

The paper reports binarized AUROC = 0.86. Your task is ordinal Spearman/Pearson. These are not directly comparable. To get a calibrated reference number, **run the Gaborieau feature set through your ordinal harness yourself**, on your folds, and use that as the baseline-to-beat. This is exactly the reproduction step.

### 7.5 Metric primer: AUROC, AUPR, Pearson, Spearman

Definitions and intuition for the four metrics we use to evaluate reproduction and the brief's task. All four take a vector of predictions $\hat{y}$ and a vector of true labels $y$.

**AUROC** (Area Under the Receiver Operating Characteristic curve). Used for binary classification. Two equivalent definitions:

1. **Geometric.** Sweep the classifier's decision threshold from 1 (predict everything negative) down to 0 (predict everything positive). At each threshold, plot TPR (true positive rate = recall on positives) on the y-axis vs FPR (false positive rate = one minus recall on negatives) on the x-axis. The curve goes from $(0, 0)$ to $(1, 1)$. AUROC is the integral.

2. **Probabilistic** (more intuitive): $\text{AUROC} = P(\hat{y}_\text{positive} > \hat{y}_\text{negative})$ when both are drawn at random. The probability that a randomly chosen positive sample has a higher predicted probability than a randomly chosen negative sample.

Range: AUROC $\in [0, 1]$. $0.5$ = random rank ordering; $1.0$ = perfect; $0.0$ = perfectly inverted (means labels are swapped relative to your scores). AUROC is **threshold-free and rank-based** -- depends only on relative ordering of scores, not absolute values.

**AUPR** (Area Under the Precision-Recall curve). Used for binary classification, especially with rare positives. Sweep threshold the same way as AUROC, but plot precision (y-axis) vs recall (x-axis). The integral is AUPR.

Range: AUPR $\in [0, 1]$ with **a baseline that depends on positive prevalence**. A random predictor's expected AUPR equals the positive class fraction -- so a 5%-positive phage's random baseline is AUPR$=0.05$, while a 50%-positive phage's is AUPR$=0.50$. Cross-phage comparison of raw AUPR is misleading; compare against each phage's prevalence floor.

AUPR is more sensitive to performance on the positive class than AUROC. The paper uses **mean AUPR across CV folds** as the criterion for picking the best model class per phage (Sec.6.8).

**Pearson correlation** $r$. Used for real-valued predictions vs real-valued labels.

$$r = \frac{\text{cov}(X, Y)}{\sigma_X \cdot \sigma_Y}$$

Measures the **linear** relationship between X and Y. Range $[-1, 1]$. $r = 1$: perfect positive linear; $r = -1$: perfect negative linear; $r = 0$: no linear relationship. **Sensitive to scale**: if $\hat{y}$ lives in $[0, 1]$ but $y$ lives in $\{0, 1, 2, 3, 4\}$, Pearson is bounded below 1 even when the ranking is perfect -- the slope of the best linear fit can't reach 1 because the predicted range is compressed.

**Spearman correlation** $\rho$. Pearson computed on the **ranks** of $X$ and $Y$.

$$\rho = r(\text{rank}(X), \text{rank}(Y))$$

Measures the **monotonic** relationship. Range $[-1, 1]$. $\rho = 1$: perfectly rank-aligned -- any monotone-increasing transformation of either variable gives $\rho = 1$. **Robust to monotonic nonlinearities and outliers.** Doesn't care if you predict $[0.05, 0.55, 0.92, 0.97]$ against true $[0, 1, 3, 4]$ -- only the rank order matters.

| Metric | Range | Random baseline | Sensitive to | Used for |
|---|---|---|---|---|
| AUROC | $[0, 1]$ | $0.5$ | Rank order only | Binary, paper reproduction |
| AUPR | $[0, 1]$ | positive prevalence | Rank + emphasis on rare class | Binary, paper model selection |
| Pearson | $[-1, 1]$ | $0$ | Linear scale match | Ordinal regression (brief) |
| Spearman | $[-1, 1]$ | $0$ | Rank order only | Ordinal regression (brief) |

### 7.6 Binary classifier as ordinal score: the rank-vs-scale tradeoff

The paper trains a binary classifier on `y_binary = (y > 0)`, producing $P(\text{lytic}) \in [0, 1]$. The brief asks for ordinal regression on the 0-4 MLC scores. **The same trained model serves both tasks**, but with a tradeoff between Spearman and Pearson that's worth understanding before reading our reproduction numbers.

**Concrete example.** Four strains, one phage, true MLC scores $[0, 1, 3, 4]$. The binary classifier sees binarized labels $[0, 1, 1, 1]$ and produces illustrative probabilities $[0.05, 0.55, 0.92, 0.97]$.

| Metric | Comparison | Result |
|---|---|---|
| AUROC vs `y_binary` | rank of $[0.05, 0.55, 0.92, 0.97]$ vs $[0, 1, 1, 1]$ | $1.0$ (negative ranked below all positives) |
| Spearman vs `y_ordinal` | rank of $[0.05, 0.55, 0.92, 0.97]$ vs $[0, 1, 3, 4]$ | $1.0$ (orders match exactly) |
| Pearson vs `y_ordinal` | raw $[0.05, 0.55, 0.92, 0.97]$ vs $[0, 1, 3, 4]$ | $\approx 0.92$ (capped by saturation) |

**Why Spearman = 1 but Pearson < 1.** The probabilities are monotone-aligned with the true MLC (rank-wise), but the relationship isn't linear. The MLC gap from 3 to 4 maps to $P(\text{lytic})$ going from $0.92$ to $0.97$ (delta $0.05$), while the MLC gap from 0 to 1 maps to $0.05 \to 0.55$ (delta $0.50$). The classifier saturates near 1 for high-MLC strains, compressing the gap between "potently lytic" and "modestly lytic." Spearman doesn't care; Pearson penalizes the nonlinearity directly.

**Information loss.** The Spearman-vs-Pearson gap is the *scale-mismatch penalty* for using a binary classifier on an ordinal task. The classifier was never told to distinguish MLC 1 from MLC 4 -- those got the same binary training label -- so its output has no incentive to differentiate them at the upper end. Pearson punishes this; Spearman doesn't.

**Practical consequence for reproduction.** When we run the paper's binary classifiers through the brief's ordinal metrics, expect:

- Within-host and within-phage **Spearman** to look quite good (the classifier preserves rank well within each phage column / strain row).
- Within-host and within-phage **Pearson** to look lower than Spearman, because the linear relationship between $P(\text{lytic})$ and the 0-4 MLC scale is compressed by the $[0, 1]$ codomain.

That Spearman-Pearson gap is the *floor of improvement available from native ordinal regression* (Sec.7.7) -- a regressor trained on the integer score could in principle close it by predicting on the actual 0-4 scale.

### 7.7 Ordinal regression options

The brief's task is ordinal regression on 0-4 MLC scores. Four approaches, in increasing methodological commitment:

| Approach | Loss | Output | Pros | Cons |
|---|---|---|---|---|
| (1) Binary + $P(\text{lytic})$ | binary cross-entropy on `(y > 0)` | scalar in $[0, 1]$ | Faithful to paper; sklearn-native | Caps Pearson via scale mismatch; ignores within-positive ordering |
| (2) Regression on integer | MSE on `y` as real | real-valued scalar | Simple; uses ordinality | Assumes smooth/linear underlying; ignores 0-4 bounds |
| (3) Multiclass + expected value | multiclass cross-entropy on `y` as 5-way label; predict $E[k] = \sum_k k \cdot P(y = k)$ | scalar in $[0, 4]$ | Closest to "5-class from a distribution" framing | Loss ignores label ordering (see below) |
| (4) Ordinal regression | cumulative-link / proportional-odds, ordinal-aware loss | class label or scalar | Formally correct ordinal | More setup; not native to sklearn (use `statsmodels.OrderedModel` or `mord`) |

**Why approach (3) requires more than just vanilla multiclass.** The conceptual framing -- each cell of the 402x96 matrix is a sample from a 5-category distribution conditional on features, train multiclass classifier, take expected value -- is sound. The catch is that **vanilla multiclass cross-entropy treats the 5 labels as unordered categories**. It penalizes the predicted probability of the true class being low; it is indifferent to where the *rest* of the probability mass goes.

Concrete example with true label = 3:

| Prediction | $p_3$ | Cross-entropy loss | Expected value | Ordinal quality |
|---|---|---|---|---|
| A: $[0.0, 0.0, 0.0, 0.5, 0.5]$ | $0.5$ | $-\log(0.5)$ | $3.5$ | Off by $0.5$ |
| B: $[0.5, 0.0, 0.0, 0.5, 0.0]$ | $0.5$ | $-\log(0.5)$ | $1.5$ | Off by $1.5$ |

Both predictions have identical cross-entropy loss -- both got $p_3 = 0.5$. Vanilla multiclass training has no gradient signal preferring A over B. Whichever the model converges on depends on regularization and other forces, not on ordinal correctness.

To restore ordinal awareness, swap cross-entropy for an **ordinal-aware loss**:

- **Earth Mover's Distance** (Wasserstein) between predicted distribution and the one-hot at the true class. Predicting mass near $k$ costs less than predicting mass far from $k$.
- **Squared-distance-weighted cross-entropy**, e.g. $\sum_k (k - y)^2 \cdot p_k$.
- **Cumulative-link logistic** (proportional-odds model): predict $P(y \leq k \mid x)$ for each threshold $k$, constrained monotone non-decreasing in $k$.

These let the classifier exploit the label ordering during training, not just during evaluation.

**Empirical caveat.** In practice, vanilla multiclass + EV often works "okay" even without ordinal-aware loss, because the underlying biology gives the classifier enough signal that nearby-class probabilities come out sensibly without explicit constraint. The features distinguishing class 0 from class 4 are different from those distinguishing class 0 from class 1, so the trained probability mass tends to cluster sensibly. Don't expect optimal results, but it's not worthless as a baseline.

**Practical recommendation for going beyond reproduction.** Run (1)-(4) as parallel arms on the same CV folds, report all four under each of the four metric variants (Sec.7.1). The win from native ordinal regression over binary + probabilities is the *empirical* test of whether respecting ordinality matters for this dataset. Prior intuition says "yes, marginally"; empirical answer is what matters.

---

## 8. The overfitting cliff

n = 402 effective samples (~250-350 after grouping near-clones).

| Representation | Dim | n/d | Safe? |
|---|---|---|---|
| Gaborieau hand-crafted (one-hot expanded) | ~120-150 | 2-3:1 | yes (with regularization) |
| 7-mer counts | 16,384 | 0.02:1 | no |
| Full pan-genome PA (Panaroo) | 15k-30k | 0.02:1 | no |
| sourmash + PCoA-32 | 32 | 12:1 | yes |
| ESM-2 mean-pool over all proteins | 1280 | 0.3:1 | no |
| ESM-2 per OMP receptor, PCA-3 each, ~10 OMPs | ~30 | 13:1 | yes |
| VLAD over rfb cluster, K=32, PCA-16 | 16 | 25:1 | yes |
| All above ESM-2/VLAD additions combined | ~80 | 5:1 | yes |
| Hand-crafted + all ESM-2 additions | ~200 | 2:1 | borderline (heavy L1 needed) |

**Rule of thumb:** anything > ~150 final dims needs heavy L1, per-fold PCA, or biology-targeted dimensionality reduction.

### 8.1 Three reduction mechanisms used in the paper

1. **Biology-guided manual feature selection** (Gaborieau's 6 chosen features).
2. **Rare-category binning** for O-type and ST_Warwick.
3. **L1 regularization** during training (additional implicit selection).

PCA on raw features is NOT used in the paper -- only UMAP on the phylogenetic distance matrix.

---

## 9. Learned representations -- what to add to the hand-crafted baseline

**Read Sec.1.1 again before starting any of this.** None of these should be attempted until reproduction is validated.

### 9.1 Why beating hand-crafted is hard but possible

Hand-crafted features encode decades of microbiology about which surface molecules matter. A learned representation won't "discover" new biology from 402 examples. But the categorical encoding has specific identifiable holes:

1. **Within-category variation** -- two strains both "O157" are encoded identically even if their *waaL* differs by 20 amino acids.
2. **Untypeable strains** (~10-25% per ECTyper benchmarks) -- all collapsed to one "Other" label even though they have different novel/hybrid clusters.
3. **Phylogeny encoded three times** (UMAP + ST + ST_host) wastes ~50 dims on largely redundant signal -- could potentially be compressed.
4. **OMPs and other receptor proteins are not used** -- their cluster IDs were extracted (sitting in `outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv` as cluster assignments per strain x OMP) but never went into the predictor.

Expected gain: small but defensible, **order of 0.02-0.08 Pearson**, concentrated on untypeable / rare-serotype strains.

### 9.2 Two complementary ESM-2 strategies

**For single-copy receptor proteins** (every strain has exactly one):
- Mean-pool ESM-2 -> 480-dim -> per-fold PCA-3 -> concatenate.
- Targets (12 OMPs from Fig 1H / repo TSV plus WaaL): OmpA, OmpC, OmpF, LamB, FhuA, BtuB, NfrA, LptD, TolC, Tsx, YncD, FadL, WaaL.
- 13 proteins x 3 components = ~39 added dims.
- Easy, well-defined, low-risk. **Recommended.**

**For variable-cardinality clusters** (different gene composition per strain):
- VLAD over ESM-2 embeddings -> K x 480 raw -> per-fold PCA -> concatenate.
- Targets: *rfb* cluster, *kps* cluster, *cps/wba* cluster.
- ~3 clusters x 16 dims = ~48 added dims.
- Harder (extraction is delicate; codebook fits per-fold). **Aspirational for 6h.**

### 9.3 Why VLAD/Fisher, not mean-pool, for variable-cardinality clusters

A typical *rfb* cluster contains 3-6 glycosyltransferases (paralogs of each other, ~25-45% identity), plus *wzx*, *wzy*, and sugar biosynthesis enzymes -- 10-25 proteins per strain depending on O-type.

Mean-pooling ESM-2 embeddings across the whole cluster assumes proteins are exchangeable. They aren't -- a strain with five GT-A-fold glycosyltransferases makes a fundamentally different sugar than a strain with five GT-B-fold glycosyltransferases, but their averaged embeddings might look nearly identical. The average destroys cluster composition.

**VLAD (Vector of Locally Aggregated Descriptors)** solves this by:
1. Fitting a codebook of K cluster centers via K-means on all training-strain proteins from this cluster type.
2. For each strain, summing residuals `(p - c_k)` for each protein `p` assigned to each center `c_k`.
3. Result: a K x d vector encoding "how many proteins of each type does this strain have, and how do they differ from the centroids?"

**Fisher vectors** are the more sophisticated version using a Gaussian Mixture Model instead of K-means, adding second-order (variance) statistics. Roughly 2x the output dimensionality. For 6 hours, VLAD is the right choice.

### 9.4 Sizing K and PCA -- the key design decision

K controls codebook granularity. PCA controls output dimensionality. **They are independent knobs.**

Bounds on K for the *rfb* cluster (~5,000 training proteins pooled across ~350 strains, ~30 effective O-type categories):

- **Lower bound: ~$\log_2(\mathrm{categories})$** -- need enough centers that distinct types separate.
- **Upper bound: ~categories**, or in practice ~50-100 proteins per center for K-means stability -- so $K \leq$ ~50 for *rfb*.
- **Image-retrieval convention: K=256+ with millions of training descriptors.** Doesn't apply here.

**Recommended K per cluster:**
- *rfb* cluster (~30 O-type categories): K=32
- *waa* operon (~6 LPS-type categories): K=8 to K=16
- *kps* cluster (~14 ABC categories): K=16
- *cps/wba* cluster (~11 Klebs categories): K=16

**Recommended PCA target:** match the categorical encoding's effective dimensionality, or slightly less. For *rfb* with ~30 one-hot dims, PCA to ~16 components.

### 9.5 Per-fold discipline (non-negotiable)

The K-means codebook AND the PCA must be fit on training-fold data only, then applied to test-fold strains. Fitting either on all 402 strains leaks information.

```python
# Inside the CV loop:
codebook = KMeans(n_clusters=K).fit(np.vstack([p for s in train_strains for p in s.proteins]))
train_vlads = np.stack([vlad_encode(s.proteins, codebook) for s in train_strains])
pca = PCA(n_components=N).fit(train_vlads)
# Now transform both train and test
train_features = pca.transform(train_vlads)
test_vlads = np.stack([vlad_encode(s.proteins, codebook) for s in test_strains])
test_features = pca.transform(test_vlads)
```

Use `sklearn.pipeline.Pipeline` to enforce this.

### 9.6 Don't ESM-embed MLST genes

The 7 housekeeping genes (*adk*, *fumC*, *gyrB*, *icd*, *mdh*, *purA*, *recA*) have no mechanistic link to phage tropism. Their sequence variation is already captured by UMAP-8. ESM-embedding them is mostly redundant dimensions.

---

## 10. The ESM-2 pipeline -- install and run

Three stages: **annotate genome -> extract specific genes/clusters -> embed with ESM-2**. ESM-2 itself just takes amino-acid sequences in and outputs vectors; it doesn't know about genomes.

### 10.1 Stage 1 -- Genome annotation

**Install Bakta (more thorough) or Prokka (faster):**
```bash
conda create -n genomics python=3.10
conda activate genomics

# Option A: Bakta
conda install -c bioconda bakta
bakta_db download --output ~/bakta_db --type full   # ~30 GB
# or smaller:
# bakta_db download --output ~/bakta_db --type light  # ~2 GB

# Option B: Prokka (no separate DB download)
conda install -c bioconda prokka
```

**Run on one genome:**
```bash
bakta --db ~/bakta_db --output bakta_out/strain_001 \
      --prefix strain_001 --threads 8 genomes/strain_001.fna
```

**Run on all 402 in parallel:**
```bash
ls genomes/*.fna | parallel -j 4 \
  'bakta --db ~/bakta_db --output bakta_out/{/.} --prefix {/.} --threads 8 {}'
```

**Wall-clock:** Bakta ~7 min/genome on 8 cores -> ~6 hours total on 32 cores running 4 in parallel. Prokka is ~2x faster.

**This consumes the entire 6-hour onsite budget.** Pre-annotate before the onsite. There's no easy fallback: the repo's `outer_membrane_proteins/` directory has only cluster IDs (a wide TSV), NOT pre-extracted protein sequences -- you can't ESM-embed cluster IDs. If you can't pre-annotate, you're limited to features that don't require sequence (hand-crafted categoricals, ANI sketches, Mash sketches, etc.) or to using the cluster IDs themselves as categorical features.

**Output per genome:**
```
bakta_out/strain_001/
+-- strain_001.gff3   # gene locations
+-- strain_001.faa    # all predicted protein sequences (FASTA)
+-- strain_001.ffn    # all predicted gene sequences (FASTA, nucleotide)
+-- strain_001.gbff   # GenBank format
\-- strain_001.tsv    # tab-separated annotation summary
```

The `.tsv` has columns: `Sequence Id, Type, Start, Stop, Strand, Locus Tag, Gene, Product`.

### 10.2 Stage 2a -- Extract single-copy receptor proteins

```python
import pandas as pd
from Bio import SeqIO
from pathlib import Path

RECEPTOR_GENES = [
    # The 12 OMPs in paper Fig 1H and the repo's clustering TSV
    "ompA", "ompC", "ompF", "lamB", "fhuA",
    "btuB", "nfrA", "lptD", "tolC", "tsx",
    "yncD", "fadL",
    # Plus WaaL (LPS O-antigen ligase, single-copy, useful for within-LPS-type variation)
    "waaL",
]

def extract_receptor_proteins(bakta_dir: Path, genome_id: str) -> dict[str, str]:
    """Return {gene_name: amino_acid_sequence} for receptors found in this genome."""
    tsv_path = bakta_dir / genome_id / f"{genome_id}.tsv"
    faa_path = bakta_dir / genome_id / f"{genome_id}.faa"
    
    annotations = pd.read_csv(tsv_path, sep="\t", comment="#")
    
    gene_to_locus = {}
    for gene in RECEPTOR_GENES:
        matches = annotations[annotations["Gene"] == gene]
        if len(matches) >= 1:
            gene_to_locus[gene] = matches.iloc[0]["Locus Tag"]
    
    locus_to_seq = {}
    for record in SeqIO.parse(faa_path, "fasta"):
        locus_to_seq[record.id] = str(record.seq)
    
    return {gene: locus_to_seq[locus] for gene, locus in gene_to_locus.items() 
            if locus in locus_to_seq}
```

### 10.3 Stage 2b -- Extract variable-cardinality cluster proteins

For the *rfb* cluster, use positional extraction between conserved flanking markers (*galF* upstream, *gnd* downstream):

```python
def extract_rfb_cluster(bakta_dir: Path, genome_id: str) -> list[str]:
    """Extract amino acid sequences of all proteins in the rfb cluster.
    
    Returns: variable-length list of protein sequences (10-25 per strain typically).
    Returns empty list if galF or gnd missing or on different contigs.
    """
    tsv_path = bakta_dir / genome_id / f"{genome_id}.tsv"
    faa_path = bakta_dir / genome_id / f"{genome_id}.faa"
    
    annotations = pd.read_csv(tsv_path, sep="\t", comment="#")
    
    galf = annotations[annotations["Gene"] == "galF"]
    gnd = annotations[annotations["Gene"] == "gnd"]
    
    if len(galf) == 0 or len(gnd) == 0:
        return []
    
    galf_contig = galf.iloc[0]["Sequence Id"]
    gnd_contig = gnd.iloc[0]["Sequence Id"]
    if galf_contig != gnd_contig:
        return []  # cluster spans contigs
    
    cluster_start = min(galf.iloc[0]["Start"], gnd.iloc[0]["Start"])
    cluster_end = max(galf.iloc[0]["Start"], gnd.iloc[0]["Start"])
    
    cluster_genes = annotations[
        (annotations["Sequence Id"] == galf_contig) &
        (annotations["Start"] > cluster_start) &
        (annotations["Stop"] < cluster_end) &
        (annotations["Type"] == "cds")
    ]
    locus_tags = set(cluster_genes["Locus Tag"])
    
    sequences = []
    for record in SeqIO.parse(faa_path, "fasta"):
        if record.id in locus_tags:
            sequences.append(str(record.seq))
    return sequences
```

**Cluster boundary markers for other clusters:**
- *waa* operon: search for genes whose name matches `waa*` in a contiguous region (no clean flanking markers).
- *kps* cluster: bracket by *kpsM*/*kpsT* (conserved ABC transporter components).
- *cps* cluster: use Kaptive's reported locus boundaries directly.

### 10.4 Stage 3 -- Run ESM-2

**Install:**
```bash
pip install torch transformers biopython
# For GPU:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Embed proteins (batched):**
```python
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"   # 480-dim output, fast
# Alternatives: "facebook/esm2_t30_150M_UR50D" (640-dim), 
#               "facebook/esm2_t33_650M_UR50D" (1280-dim)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

@torch.no_grad()
def embed_proteins_batched(sequences: list[str], batch_size: int = 16) -> torch.Tensor:
    """Return (N, d) tensor of mean-pooled embeddings."""
    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=1024).to(device)
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state  # (B, L, d)
        
        # Mean-pool ignoring padding, <cls> at start, <eos> at end
        mask = inputs.attention_mask.unsqueeze(-1).float()
        mask[:, 0] = 0  # drop <cls>
        seq_lens = inputs.attention_mask.sum(dim=1)
        for j, length in enumerate(seq_lens):
            mask[j, length-1] = 0  # drop <eos>
        
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0)
```

**Compute cost:** ~5,000 proteins (avg ~250 aa) on T4 GPU: 2-3 minutes. On CPU: 30-60 minutes. **Use a GPU if available.**

### 10.5 Stage 4 -- Build VLAD over the embeddings

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class VLADFeaturizer:
    """VLAD encoding for variable-cardinality protein sets per strain.
    
    Fits codebook (KMeans) and PCA on training fold only.
    """
    def __init__(self, k_codebook: int = 32, n_pca: int = 16, random_state: int = 0):
        self.k = k_codebook
        self.n_pca = n_pca
        self.random_state = random_state
        self.kmeans = None
        self.pca = None
    
    def fit(self, train_proteins_per_strain: list[np.ndarray]):
        """train_proteins_per_strain[i] = (N_i, d) embeddings for strain i."""
        all_proteins = np.vstack(train_proteins_per_strain)
        
        self.kmeans = KMeans(n_clusters=self.k, n_init=10, 
                             random_state=self.random_state).fit(all_proteins)
        
        raw_vlads = np.stack([self._encode_one(p) for p in train_proteins_per_strain])
        self.pca = PCA(n_components=self.n_pca, 
                       random_state=self.random_state).fit(raw_vlads)
        return self
    
    def _encode_one(self, protein_embeddings: np.ndarray) -> np.ndarray:
        d = self.kmeans.cluster_centers_.shape[1]
        if len(protein_embeddings) == 0:
            return np.zeros(self.k * d)
        
        assignments = self.kmeans.predict(protein_embeddings)
        centers = self.kmeans.cluster_centers_
        
        vlad = np.zeros((self.k, d))
        for k in range(self.k):
            mask = assignments == k
            if mask.sum() > 0:
                vlad[k] = (protein_embeddings[mask] - centers[k]).sum(axis=0)
        
        vlad = vlad.flatten()
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))  # power normalization
        norm = np.linalg.norm(vlad)
        if norm > 0:
            vlad = vlad / norm
        return vlad
    
    def transform(self, proteins_per_strain: list[np.ndarray]) -> np.ndarray:
        raw_vlads = np.stack([self._encode_one(p) for p in proteins_per_strain])
        return self.pca.transform(raw_vlads)
```

**Wrap this in your `Featurizer` API.** The `fit` method gets called with training-fold proteins only; `transform` is called for both train and test.

### 10.6 End-to-end pipeline diagram

```
402 FASTA files
    |
    v Bakta (annotation, ~6h wall-clock on 32 cores -- DO BEFORE ONSITE)
402 directories with .tsv + .faa per genome
    |
    v Custom Python (per-strain receptor/cluster extraction; ~5 min total)
402 dicts {gene: sequence} + 402 lists [cluster_seqs]
    |
    v ESM-2 batched (~5 min for OMPs, ~10 min for clusters, on GPU)
402 x d embedding per receptor + 402 lists of (N_i, d) cluster matrices
    |
    v Per-fold: KMeans codebook + VLAD encoding (clusters) / PCA (receptors)
402 x n_pca features per receptor type / cluster, refit per CV fold
    |
    v Concatenate with hand-crafted ~120-150 dim baseline
402 x (~150 + ~80) input matrix -> per-phage Ridge/Logistic/RF
```

---

## 11. Realistic 6-hour onsite plan (reproduction-first)

Assuming pre-onsite prep from Sec.12 is done.

### Stage 0: Setup (Hour 0.0-0.5)

- Ask the open questions in Sec.13.
- Load data, sanity-check shapes.
- Confirm `picard_collection.csv` has the expected columns.

### Stage 1: REPRODUCE THE PAPER (Hour 0.5-3.0)

**This is the priority. Do not start Stage 2 until Stage 1 is validated.**

- Build eval harness: 10-fold GroupKFold from skani ANI $\geq 99.99\%$, four metrics, no-model baseline.
- Sanity-check: no-model baseline produces high within-host Pearson, ~0 within-phage.
- Implement the exact paper feature pipeline:
  - Read `picard_collection.csv` and `coli_umap_8_dims.tsv`.
  - Apply regex filter `(UMAP|O-type|LPS|ST_Warwick|Klebs|ABC_serotype)`.
  - Merge phage host features (ST_host, H_host).
  - Compute the four `same_*_as_host` booleans (including the bug -- match the script exactly).
  - <3 binning for O-type and ST_Warwick only.
  - One-hot encode, standardize UMAP.
- Train the four per-phage models with the paper's class weights.
- Select best-per-phage by mean AUPR.
- **Validation checkpoint:** your per-phage AUROCs should be in the ballpark of `dev/predictions/per_phage_perf.csv`. If they're off by more than ~0.05 AUROC on average, there's a bug; debug before proceeding.
- Compute Spearman/Pearson on ordinal 0-4 (your actual task). This is your **calibrated reference number**.

### Stage 2: One novel representation (Hour 3.0-5.0)

**Only if Stage 1 is validated.** Pick ONE of the following based on time remaining:

- **Easy: cluster-ID categorical baseline.** Use the shipped `outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv` directly: 12 columns of cluster IDs per strain x OMP. One-hot encode, concatenate to hand-crafted. ~few hundred added dims after rare-cluster binning. No annotation needed. Honest first pass at "does OMP identity carry signal beyond hand-crafted features?"
- **Medium: ESM-2 on extracted receptors (requires annotation).** Run Bakta/Prokka first (pre-onsite), extract the 12 OMPs by gene name, embed with ESM-2, per-fold PCA-3 per protein, concatenate. ~36 added dims.
- **Hard: VLAD on *rfb* cluster.** Requires Bakta annotations done in advance. K=32, PCA-16. Mention as a future direction if not attempted.

### Stage 3: Stratified analysis (Hour 5.0-5.5)

- Where do errors live? Which strains do models disagree on?
- Stratify performance on typed vs untypeable strains.
- If learned representation helps mainly on untypeable strains, the mechanistic story holds. If gains are uniform, something else is going on.

### Stage 4: Write-up (Hour 5.5-6.0)

- One table: methods x four metrics. One row per featurizer.
- One sentence per row: what it captures biologically, did it help, why.
- Honest list of what to try next.

### Things to skip entirely

- Full Panaroo run (use UMAP-8 directly from the repo).
- DefenseFinder elaborate engineering (paper showed marginal contribution).
- Genomic LMs at whole-genome scale (Evo, NT, DNABERT -- context too small or compute too large).
- Joint multi-output shared-encoder (interesting future work; mention in write-up).
- Featurizing phages (out of scope per task framing).
- Fixing the `same_ABC_as_host` bug as part of reproduction (note it, don't fix it during reproduction).

---

## 12. Pre-onsite prep (highest-EV preparation)

### Evening 1 (~2-3h) -- Get reproduction tooling working

1. Clone the repo:
   ```bash
   git clone https://github.com/mdmparis/coli_phage_interactions_2023.git
   cd coli_phage_interactions_2023
   ```
2. Verify expected files exist: `picard_collection.csv` (403 rows), `coli_umap_8_dims.tsv`, `interaction_matrix.csv` (402 rows), `outer_membrane_proteins/blast_results_cured_clusters=99_wide.tsv`, `defense_finder/`, `panacota/tree/`, `isolation_strains/panacota/tree/370+host_distance_matrix.tsv`.
3. Load data into pandas. Confirm shapes ($402 \times 96$ matrix, 403 rows in feature CSV -- one strain has metadata but no interactions; see Sec.2.4).
4. Write the eval harness with a placeholder featurizer (returns random vectors). Verify the four metrics compute and the no-model baseline behaves asymmetrically as predicted.

### Evening 2 (~3-4h) -- Reproduce the paper baseline

1. Implement the exact Sec.6 feature pipeline. Match the script line-by-line including the bug.
2. Compute CV folds from the Newick tree at `data/genomics/bacteria/panacota/tree/`, threshold $<10^{-4}$, group strains.
   - If parsing the tree is annoying, use `skani triangle` with ANI $\geq 99.99\%$ as a proxy.
3. Train per-phage models, compare per-phage AUROCs against `dev/predictions/per_phage_perf.csv`.
4. **If you can't reproduce within ~0.05 AUROC on average, fix it now, not during the onsite.**

### Evening 3 (~3-4h, optional but valuable) -- Get ESM-2 working

1. Fresh conda env with PyTorch + transformers + biopython.
2. Download `facebook/esm2_t12_35M_UR50D` (small enough to be quick).
3. Verify embedding works on a single test sequence.
4. Annotate genomes (Bakta or Prokka), extract OMP sequences by gene name from the `.faa` outputs, and run ESM-2 on them. Save the embeddings to disk so you don't have to recompute during the onsite. (Note: the shipped `outer_membrane_proteins/` directory has only cluster IDs, not sequences -- you must annotate the FASTAs to get amino-acid strings.)

### Optional Evening 4 -- Bakta annotation

Only if you want VLAD-on-clusters available during the onsite.

1. Install Bakta + download DB.
2. Run on all 402 genomes overnight.
3. Verify outputs.

---

## 13. Open questions to ask Tabula at the start

In rough priority order:

1. **"What are the additional pre-computed genome representations you mentioned?"** Could include Mash sketches, ESM embeddings, defense vectors. Reshapes priorities.
2. **"Are the phage genomes available in the data drop, and is featurizing phages in scope?"** Confirms task framing.
3. **"For computing CV folds since the original group file isn't in the repo, do you have a preferred method?"**
4. **"For the ordinal regression vs binary task -- should I also report binarized AUROC for direct comparison to the paper?"**
5. **"What's the compute setup -- GPU available? Internet access for Hugging Face / Bakta?"**
6. **"How important is reproducing Gaborieau's exact numbers vs trying something new?"** Confirms reproduction-first priority.

---

## 14. Things to verify (items I'm uncertain about)

Flagged for the agent to confirm by reading code/data:

1. **Exact one-hot dimensionality after binning.** Estimated 120-150 but haven't run the code. Resolve by running.
2. **Score aggregation rule** from raw replicate scores to 0-4 integer. Check `data/interactions/raw/raw_interactions.csv` vs `interaction_matrix.csv`.
3. **Strain count (resolved).** Paper text: 403 strains, 38,688 interactions. Strain catalog `picard_collection.csv`: 403 rows. Published `interaction_matrix.csv`: **402 rows** (one strain has metadata but no interaction labels). Tabula ships the 402-row matrix. Treat working data as 402; cite paper-level claims as 403. See Sec.2.4 for the full picture.
4. **What's in `outer_membrane_proteins/` (resolved).** A single wide TSV `blast_results_cured_clusters=99_wide.tsv` listing one cluster ID per strain x OMP at 99% identity. 12 OMP columns, upper-case: BTUB, FADL, FHUA, LAMB, LPTD, NFRA, OMPA, OMPC, OMPF, TOLC, TSX, YNCD. Matches Fig 1H. (FepA isn't in either.)
5. **Tabula's "additional pre-computed representations"** -- ask at start.
6. **Whether phage genomes are in the data drop** -- ask at start.
7. **CV grouping reconstruction.** Compare fold sizes against the paper's Figure 5 numbers.

---

## 15. Corrections to earlier guidance / mistakes I made

These are reminders of false claims from prior iterations of this document and the conversation it was distilled from. Don't repeat them:

1. **"~50 dimensions" was wrong.** Actual post-one-hot is ~120-150.
2. **"Genomes are ~40k bp" was wrong.** Actual ~5 Mbp.
3. **The README's 6-feature list omits engineered features.** ST_host and four `same_*_as_host` booleans are also in the actual feature vector. (Earlier doc iterations claimed H_host was also present; it isn't -- the H-type column is dropped by the line-57 regex filter before the line-79 rename runs, so the H-type -> H_host rename is a silent no-op. See Sec.6.2.)
4. **`same_ABC_as_host` is buggy** (compares column to itself). Reproduce the bug for faithful reproduction; note it for honest discussion.
5. **K=8 for VLAD was arbitrary.** The principled K is matched to biological diversity of the cluster being encoded -- typically K=16 to K=32 for the polysaccharide clusters.
6. **Naive concatenation of 480-dim ESM-2 to hand-crafted will overfit.** Always per-fold PCA, always restricted to ~3-16 components per protein/cluster.
7. **Don't ESM-embed MLST genes.** They're housekeeping enzymes with no phage relevance.
8. **Mean-pooling ESM-2 across variable-cardinality clusters destroys composition information.** Use VLAD/Fisher.
9. **The "6-hour hour-by-hour plan" in v1 was unrealistic** -- it assumed pre-onsite prep done. v2's reproduction-first plan is more honest.

---

## 16. Code skeleton

```python
# featurizers.py -- pluggable feature extractors

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class Featurizer(ABC):
    """Abstract base for all genome featurizers.
    
    Implementations must take a list of genome IDs (matching the bacterial
    strain index of picard_collection.csv) and return a 2D numpy array of
    shape (n_genomes, n_features).
    
    Must support per-fold fitting to avoid leakage.
    """
    @abstractmethod
    def fit(self, train_genome_ids: list[str]): ...
    
    @abstractmethod
    def transform(self, genome_ids: list[str]) -> np.ndarray: ...
    
    def fit_transform(self, train_genome_ids: list[str]) -> np.ndarray:
        self.fit(train_genome_ids)
        return self.transform(train_genome_ids)


class GaborieauHandCrafted(Featurizer):
    """Faithful reproduction of the paper's six-feature pipeline.
    
    Reads picard_collection.csv + coli_umap_8_dims.tsv, applies the regex filter,
    merges phage-isolation-host features, computes same_*_as_host booleans
    (including the same_ABC_as_host bug exactly), applies <3-occurrence binning
    to O-type and ST_Warwick (fit on training fold), one-hot encodes, standardizes UMAP.
    """
    # ... implementation faithfully reproducing predict_all_phages.py ...


class ESM2ReceptorPCA(Featurizer):
    """ESM-2 mean-pool of one specific receptor protein, per-fold PCA.
    
    Pulls amino acid sequences from Bakta/Prokka annotations (the shipped
    outer_membrane_proteins/ TSV has only cluster IDs, not sequences).
    """
    def __init__(self, protein_name: str, n_components: int = 3): ...


class VLADClusterFeaturizer(Featurizer):
    """VLAD encoding for variable-cardinality cluster proteins, per-fold codebook + PCA.
    
    Uses Sec.10.5 VLAD implementation; requires Bakta-annotated genomes.
    """
    def __init__(self, cluster_extractor: callable, k_codebook: int = 32, n_pca: int = 16): ...


class Concatenated(Featurizer):
    """Concatenate multiple featurizers' outputs. fit/transform delegate appropriately."""
    def __init__(self, *featurizers: Featurizer): ...


# eval.py

def grouped_kfold_from_ani(ani_matrix_path: str, threshold: float = 99.99) -> np.ndarray:
    """Compute connected components on ANI >= threshold, return group labels."""
    ...

def evaluate(featurizer: Featurizer, X_genomes: list[str], Y: np.ndarray,
             groups: np.ndarray, model_class=None, n_splits: int = 10) -> dict:
    """Run 10-fold group CV, return all four metrics per fold."""
    ...


# baselines.py

class PerPhageMeanBaseline:
    """The 'no-model' baseline. High within-host correlation, ~0 within-phage."""
    def fit(self, X, Y): self.phage_means = Y.mean(axis=0)
    def predict(self, X): return np.tile(self.phage_means, (len(X), 1))


# main.py

if __name__ == "__main__":
    Y = pd.read_csv("data/interactions/interaction_matrix.csv", index_col=0, sep=";").values
    genome_ids = list(...)
    groups = grouped_kfold_from_ani("ani_matrix.txt")
    
    # REPRODUCTION FIRST
    featurizers_stage_1 = {
        "no_model": PerPhageMeanBaseline(),
        "gaborieau_reproduction": GaborieauHandCrafted(),
    }
    
    # Only after stage 1 is validated:
    featurizers_stage_2 = {
        "gaborieau + ompA_esm": Concatenated(
            GaborieauHandCrafted(),
            ESM2ReceptorPCA("ompA", n_components=3),
        ),
        "gaborieau + all_omp_esm": Concatenated(
            GaborieauHandCrafted(),
            *[ESM2ReceptorPCA(p, n_components=3) for p in RECEPTOR_GENES],
        ),
        # ... VLAD if time permits ...
    }
    
    results = {}
    for name, featurizer in featurizers_stage_1.items():
        results[name] = evaluate(featurizer, genome_ids, Y, groups)
    
    # Validation checkpoint: gaborieau_reproduction should match paper's per_phage_perf.csv
    assert reproduction_validates(results["gaborieau_reproduction"]), "Fix reproduction before extending"
    
    for name, featurizer in featurizers_stage_2.items():
        results[name] = evaluate(featurizer, genome_ids, Y, groups)
    
    # Format as one big comparison table
```

### Validation checkpoints

1. **Eval harness works** -- no-model baseline produces high within-host Pearson, ~0 within-phage.
2. **CV grouping looks right** -- near-clones always co-fold; ANI distribution within-fold > across-fold.
3. **Reproduction matches paper** -- per-phage AUROCs within ~0.05 of `dev/predictions/per_phage_perf.csv`.
4. **ESM-2 features sensible** -- strains with same O-type have more similar WaaL embeddings than across-O-type strains (t-SNE colored by O-type).

---

## 17. Quick reference: key numbers

| Quantity | Value | Source |
|---|---|---|
| Genomes in dataset (paper catalog) | 403 | Paper text + `picard_collection.csv` |
| Genomes in working interaction matrix | 402 | `interaction_matrix.csv` (see Sec.2.4) |
| Phages | 96 | Brief / paper |
| Total labeled scalars (paper text) | 38,688 | $403 \times 96$ (paper Fig 2C) |
| Total labeled scalars (working) | 38,592 | $402 \times 96$ in delivered CSV |
| Effective samples | 402 (~250-350 after grouping) | derived |
| Genome size | ~5 Mb | E. coli biology |
| Predicted proteins per genome | ~5,000 | Prodigal output |
| **Final pre-one-hot input columns** | **18** (13 for LF110 phages) | `[VERIFIED-BY-TRACE]` see `dev/verify_predict_pipeline.py` |
| **Final post-one-hot input dim** | **114** for normal phages, **109** for LF110 phages | `[VERIFIED-BY-TRACE]` |
| Hand-crafted feature *types* | 8 (6 bacteria + 1 phage-host + 1 boolean cluster) | `[VERIFIED-BY-TRACE]` |
| Distinct O-types in panel | up to ~93 | data |
| Distinct STs in panel | 163 | paper |
| % positive interactions | ~19-20% | paper |
| Paper headline AUROC (binarized) | 0.86 | paper |
| CV folds | 10, grouped by core-genome dist <1e-4 | README/script |
| CV grouping file status | **missing -- recompute via Newick tree or skani** | README |
| Model candidates per phage | 4 (L1, L2, RF-3, RF-6) + DummyClassifier | `[VERIFIED-FROM-CODE]` |
| Class weight schedule | 0.8 / 1 / 1.5 / 2 / 3 (by % positive) | `[VERIFIED-FROM-CODE]` |
| Expected within-host Pearson (no-model) | 0.30-0.45 | derived |
| Expected within-phage Pearson (no-model) | ~0.00 | derived |
| Target within-phage Pearson (paper) | 0.40-0.55 (extrapolation from AUROC=0.86) | educated guess |
| ESM-2 t12 dim | 480 | model card |
| ESM-2 35M compute time (all proteins) | ~3 min/cluster type on T4 GPU | estimate |
| Bakta wall-clock for 402 genomes | ~6 hours on 32 cores | estimate |
| VLAD K recommendation | K=32 for *rfb*, K=16 for *kps*/*cps*, K=8 for *waa* | matched to category counts |
| VLAD PCA target | ~16 dims per cluster | matches paper-equivalent |

---

## Appendix A: External literature pointers

- **Receptors review**: Bertozzi Silva, Storms, Sauvageau (2016) "Host receptors for bacteriophage adsorption." FEMS Microbiol. Lett. 363:fnw002.
- **Defense systems review**: Georjon & Bernheim (2023) "The highly diverse antiphage defence systems of bacteria." Nat. Rev. Microbiol. 21:686.
- **BASEL collection** (related E. coli phage interaction dataset): Maffei et al. (2021) "Systematic exploration of *E. coli* phage-host interactions with the BASEL phage collection." PLoS Biol. 19:e3001424.
- **PhageHostLearn** (Klebsiella; closest methodological cousin): Boeckaerts et al. (2024) Nat. Commun. 15:4355. The bacterial-side ESM-2 recipe is portable.
- **ESM-2**: Lin et al. (2023) "Evolutionary-scale prediction of atomic-level protein structure." Science 379:1123.
- **Bakta**: Schwengers et al. (2021) Microb. Genom. 7:000685.
- **PanACoTA**: Perrin & Rocha (2021) NAR Genom. Bioinform. 3:lqaa106.
- **skani**: Shaw & Yu (2023) Nat. Methods 20:1661.
- **DefenseFinder/MacSyFinder v2**: Neron et al. (2023) Peer Community J. 3:e28; Tesson et al. (2022) Nat. Commun. 13:2561.
- **ECTyper**: Bessonov et al. (2021) Microb. Genom. 7:000728.
- **Kaptive**: Wyres et al. (2016) Microb. Genom. 2:e000102.
- **ClermonTyping**: Beghain et al. (2018) Microb. Genom. 4:e000192.
- **VLAD**: Jegou et al. (2010) "Aggregating local descriptors into a compact image representation." CVPR.
- **Fisher Vectors**: Perronnin & Dance (2007) "Fisher kernels on visual vocabularies for image categorization." CVPR.
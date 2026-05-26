# Phage-Host Prediction in *E. coli*: A Reference for the Tabula Onsite

## Document scope and conventions

This document is a self-contained reference for a 6-hour ML research onsite at Tabula, a phage-therapy company developing treatments for antibiotic-resistant *Escherichia coli* (*E. coli*) infections. The project is multi-output regression: given a bacterial genome, predict a 96-dimensional vector of phage interaction scores.

Every biological abbreviation is defined on first use. Every feature-extraction step is described mechanically, with input, tool, what the tool actually does, and output. No prior bacterial-genomics knowledge is assumed.

Terminology note: "phage" is short for "bacteriophage," a virus that infects bacteria. "Host" in this document always means the bacterial strain. "Tropism" means which hosts a phage can infect.

---

## 1. The dataset

### 1.1 What is on disk

You receive two things.

**(a) 402 FASTA files**, one per bacterial strain.

FASTA is a plain-text file format. Each file looks like:
```
>contig_1 length=1024576
ATGCGATCGATCGATCGAT...
GGCTAGCTAGCTAGCTAGC...
>contig_2 length=523191
GTCGATCGATCGATCGAT...
```

A line starting with `>` is a header naming a sequence. The following lines are the sequence itself, in the four-letter DNA alphabet `{A, C, G, T}` (occasionally `N` for ambiguous positions). The alphabet is case-insensitive by convention but tools handle both.

Each *E. coli* genome is biologically a single circular chromosome of ~5 million base pairs (Mbp), sometimes with one or more small circular plasmids on the side. Because the genomes were assembled from short sequencing reads, you do not get one clean circle. You get **50-500 linear contigs** (contiguous pieces) that together sum to ~5 Mbp. The number of contigs is an artifact of assembly difficulty, not real biology.

Storage: $402 \times$ ~5 MB ~ **2 GB total** of raw ACGT.

**(b) An interaction matrix** of shape (402, 96).

Rows are bacterial strains, columns are phages, entries are integers in `{0, 1, 2, 3, 4}`. About 80% of entries are 0; about 20% are non-zero. Higher value = phage is effective at higher dilution = more potent infection.

The score originates from a **plaque assay**, a 100-year-old microbiology technique: you spread bacteria on agar, drop phage solutions of varying dilution on top, incubate overnight, and count cleared circles ("plaques") the next morning. A score of 4 means clear lysis even at the most dilute phage concentration tested; 0 means no lysis at any concentration. The score is approximately ordinal but not strictly linear.

That is the entire dataset. Input: 402 strings of length ~5,000,000 over `{A,C,G,T,N}`. Output: a (402, 96) integer matrix.

**Known 402-vs-403 anomaly.** The paper headline says 403 strains x 96 phages = 38,688 interactions, and the strain catalog `picard_collection.csv` in the repo has 403 rows. But the published `interaction_matrix.csv` has only **402 rows** -- one strain has metadata but no interaction labels. Tabula ships the 402-row matrix, so the *working dataset* is 402 strains; the *Picard catalog* per the paper is 403. The missing strain is not a designated holdout, just an asymmetry in the published artifacts.

### 1.2 What is held out

Standard 10-fold cross-validation (CV) across the **402 strains** (rows of the matrix). In each fold, ~362 strains are used for training and ~40 are held out for testing. For each held-out strain you predict its full 96-vector and compare to the truth.

The 96 phages are **never held out**. Column index `j` always refers to the same phage across all folds. This is important and slightly unusual:

- The model never has to generalize to a new phage.
- Phage identity is implicit in the output column index -- the model can (and should) memorize each phage's preferences.
- The only generalization being tested is: given a bacterial genome the model has never seen, can it correctly predict which of the 96 known phages will infect it?

### 1.3 Effective sample size

Naive count: $402 \times 96 = 38{,}592$ (host, phage, score) triples.

ML-effective count: **402 independent input samples**. The 96 entries for one strain all share a single input (the genome) and the same featurization, so they constitute one observation in the statistical sense.

After accounting for near-clones (see Sec.3.1 on phylogenetic leakage): probably 250-350 effectively independent rows. This is the size of a small UCI dataset. It dictates everything about model choice.

### 1.4 What "pre-computed genome representations" likely means

The brief mentions "additional pre-computed genome representations that we have been experimenting with internally." Without seeing them, the realistic candidates are: existing per-strain feature tables (e.g., a Panaroo gene presence/absence matrix), pre-computed sketches (Mash or sourmash signatures), pre-computed protein-language-model embeddings of all proteins per strain, or precomputed phylogenetic distance matrices. Ask explicitly what they are at the start of the onsite; the answer affects which of the Sec.6 representations you should prioritize implementing.

---

## 2. Biology you need to predict phage-host interaction

This section gives the minimum biology required to reason about which genome features matter and why.

### 2.1 The phage infection cascade

A phage attempting to infect a bacterium proceeds in roughly four steps. Different bacterial features control different steps.

**Step 1: Adsorption (receptor binding).** The phage uses surface proteins (tail fibers or tail spikes) to bind a specific molecule on the bacterial cell surface. If it cannot bind, nothing downstream happens. **This is the dominant determinant of host range** in diverse natural strain panels. The relevant bacterial features are surface molecules:

- **LPS** (Lipopolysaccharide). A large molecule covering most of the bacterial outer membrane. Has three parts:
  - **Lipid A**: anchor, conserved, not specificity-relevant.
  - **Core oligosaccharide**: a short sugar chain, ~10 sugars. The "outer core" portion varies between strains (R1, R2, R3, R4, K-12 types in *E. coli*).
  - **O-antigen**: a long polysaccharide of repeating sugar units, attached to the outer core by an enzyme called WaaL ("O-antigen ligase"). There are >180 known O-antigen types in *E. coli* (O1, O2, O3, ..., O157, ...). Each is made by a different cluster of biosynthesis genes called the **O-antigen locus** or *rfb* cluster, typically 10-25 genes.
- **K-antigen / capsule**. Some strains additionally have a polysaccharide capsule on top of the LPS. The biosynthesis cluster is called the *kps* locus (group 2/3 capsules) or the *cps* locus (group 1, Klebsiella-style).
- **OMPs** (Outer Membrane Proteins). $\beta$-barrel proteins embedded in the outer membrane. Specific ones are receptors for specific phages. Gaborieau et al. tracked **12 OMPs** as putative *Escherichia* phage receptors (Fig 1H). Both Fig 1H and the repo's clustering TSV use the same 12 (alphabetical in the TSV): `BTUB, FADL, FHUA, LAMB, LPTD, NFRA, OMPA, OMPC, OMPF, TOLC, TSX, YNCD`. Each is found in >97% of strains, with substantial variant diversity at 99% genomic identity. (FepA is a canonical literature receptor for *E. coli* phages but Gaborieau et al. did not include it in their tracked set; NfrA is in the slot you might otherwise expect.)
  - **LamB**: maltose porin; receptor for phage $\lambda$.
  - **OmpA, OmpC, OmpF**: porins; receptors for T-even phages and others.
  - **FhuA**: ferrichrome iron transporter; receptor for T1, T5, phage $\phi 80$.
  - **BtuB**: vitamin B12 receptor; receptor for BF23 and some others.
  - **LptD**: LPS export channel.
  - **Tsx**: nucleoside channel.
  - **TolC**: efflux outer membrane channel; receptor for some phages.
  - **YncD**: TonB-dependent receptor, function partially characterized; high variant diversity in the panel.
  - **FadL**: long-chain fatty acid transporter.
  - **NfrA**: outer-membrane receptor for phage N4.
- **Pili and flagella**. Sometimes used as receptors. The F-pilus and type IV pili are most relevant.

**Step 2: DNA injection.** The phage punctures the membrane and injects its genome. Rarely a host-specificity bottleneck once binding has occurred.

**Step 3: Intracellular replication.** Now the bacterium can fight back. Relevant features are **antiphage defense systems** -- genetic modules that detect and destroy invading phage DNA or kill the infected cell to prevent phage propagation:

- **R-M** (Restriction-Modification): enzymes that cut unmethylated foreign DNA.
- **CRISPR-Cas**: adaptive immunity -- stored memory of past phage encounters; cuts matching DNA.
- **Abortive infection** systems (Abi): cell suicide on phage detection, sacrificing the cell to protect the colony. Examples include BREX, DISARM, Druantia, Gabija, Septu, Lamassu, Hachiman.
- **CBASS** (Cyclic-oligonucleotide-Based Anti-phage Signaling System): bacterial analog of the human cGAS-STING pathway.
- **Retrons, toxin-antitoxin systems, Pycsar, Thoeris**: various other defenses.

DefenseFinder's catalog spans ~150 defense system families across all bacteria. In the *Escherichia* pan-immune system specifically, Gaborieau et al. detected **137 distinct subfamilies** across the 403-strain Picard collection (the full catalog per the paper; see Sec.1.1 for the 402/403 anomaly), with an average of ~8 subfamilies per isolate (range 1-16).

**Step 4: Lysis.** The phage finishes replicating, lyses the cell, and releases progeny. Not specificity-relevant.

### 2.2 Why surface features dominate (the empirical observation in Gaborieau et al. 2024)

The receptor binding step is a hard gate. If the phage cannot dock, nothing else matters. Across the >180 O-antigens x ~80 capsule types x variable OMP alleles in *E. coli*, the combinatorial diversity at the cell surface creates a strong specificity barrier.

Defense systems matter most in **coevolved settings**: a phage you have repeatedly encountered in the same niche. In a diverse, naive panel like Gaborieau's (96 phages tested against 403 strains per the paper, of which 402 are in the published interaction matrix; see Sec.1.1), almost all variance in infection success is explained by receptor compatibility. Gaborieau et al. explicitly state this conclusion: "Bacterial adsorption factors are the major determinants of phage-bacteria interactions in contrast with defence systems which marginally reduce virulence of infecting phages."

**Practical consequence for the project**: spend your representation-design effort on capturing surface features well. Defense systems are worth including as a one-hot vector but not worth elaborate engineering.

### 2.3 Pan-genome terminology

You will see these terms repeatedly.

- **Gene**: a stretch of DNA encoding a protein, typically 300-3000 base pairs in bacteria. *E. coli* has ~4500-5500 genes per genome.
- **Ortholog**: a gene that exists "the same" gene in multiple genomes (descended from a common ancestor, same function). E.g., the *ompA* gene exists in all *E. coli* strains; each strain's copy is an ortholog of the others.
- **Orthogroup / gene family / gene cluster**: the set of orthologous copies of one gene across a panel of genomes. One row per orthogroup in a presence/absence table.
- **Pan-genome**: the union of all gene families across all genomes in your panel. For 402 *E. coli* genomes, expect 15,000-30,000 distinct gene families.
- **Core genome**: gene families present in $\geq 99\%$ (sometimes $\geq 95\%$) of strains. For *E. coli*, ~3000-4000 families. Mostly housekeeping genes.
- **Accessory / shell / cloud genome**: gene families present in only some strains. The accessory genome is where serotype-determining clusters, defense systems, and many phage receptors actually live, and where strain-to-strain variation in phage susceptibility comes from.
- **SNP** (Single Nucleotide Polymorphism): a single-base difference between two genomes at a position where they otherwise align. Used to measure phylogenetic distance.
- **ANI** (Average Nucleotide Identity): the average percent identity of orthologous regions between two genomes. Standard measure of genome similarity. Two genomes are typically considered the "same species" at ANI $\geq 95\%$. Near-clones are ANI $\geq 99.99\%$.

### 2.4 Phylogeny

Bacterial strains can be arranged into a tree based on accumulated DNA differences. The phylogeny of *E. coli* divides into eight named "phylogroups" labeled A, B1, B2, C, D, E, F, G -- coarse clades, each with characteristic ecological associations. Within phylogroups, strains are subdivided into **sequence types** (STs) defined by allele combinations at seven housekeeping genes (the standard **MLST**, Multi-Locus Sequence Typing, scheme).

Phylogeny correlates with phage susceptibility because closely related strains tend to have similar surface molecules -- they inherited them from a common ancestor. But phylogeny is **not causal**: the actual determinants are the surface genes, which can be gained, lost, or recombined between lineages. Two strains in the same ST can differ in O-antigen due to horizontal gene transfer at the *rfb* locus. This is why models that use phylogeny *and* explicit serotype features (Gaborieau's approach) outperform models using only phylogeny.

---

## 3. Evaluation: metrics and the overfitting cliff

### 3.1 Phylogenetic data leakage -- the most important thing in this section

Random k-fold CV will give you optimistic numbers that do not reflect real generalization.

Reason: bacterial collections like the Picard panel contain near-clones -- for example, two clinical isolates from the same hospital outbreak with only ~30 single-nucleotide polymorphisms (SNPs) between them across 5 Mb. These are essentially the same genome from a feature-extraction standpoint. If one is in train and one in test, the model "generalizes" trivially.

**Mitigation: group near-clones into the same CV fold.** Gaborieau et al. used the threshold: any two strains within $10^{-4}$ substitutions per site of each other (i.e., $\leq 500$ SNPs across 5 Mb) go in the same fold. A practically equivalent and faster criterion is **ANI $\geq 99.99\%$ as computed by skani**.

Implementation sketch:

1. Compute the all-vs-all ANI matrix with skani triangle (about 1 minute for 402 genomes).
2. Threshold at 99.99%; connected components define groups.
3. Use `sklearn.model_selection.GroupKFold` with these group labels.

If you skip this, your held-out Pearson correlations will be inflated by 0.05-0.20 depending on how clonal the dataset is. The interviewers will know.

### 3.2 The four metric variants

For each fold, you produce a predicted matrix `Y_pred` of shape (n_test, 96) and compare to `Y_true` of shape (n_test, 96). Two correlation choices x two axis choices = four metrics.

**Within-host correlation** (asks: can you rank phages correctly for a given new strain?):
```python
within_host = mean(
    correlation(Y_pred[i, :], Y_true[i, :])
    for i in range(n_test)
)
```

**Within-phage correlation** (asks: can you rank strains correctly for a given phage?):
```python
within_phage = mean(
    correlation(Y_pred[:, j], Y_true[:, j])
    for j in range(n_phages)
)
```

Use both Spearman (rank correlation, robust to ordinal nonlinearity) and Pearson (assumes linear, sensitive to the spacing of the 0-4 scores).

**Average across folds and report all four numbers with standard deviations.** A single number is uninformative; the four-way table tells a story.

### 3.3 The no-model baseline and what it can do

Predict, for every held-out strain, the per-phage mean computed across the training strains:
```python
Y_pred[i, :] = Y_train.mean(axis=0)  # same prediction for every held-out i
```

What this baseline achieves:

- **Within-host correlation: substantial, possibly Pearson 0.3-0.5.** Reason: the matrix has shared structure. Some phages are broad-host-range and lyse 50% of strains; others are narrow and lyse 2%. A test strain that follows the average pattern (which most do) will have high observed scores for the broad phages and low scores for the narrow ones, matching the prediction.
- **Within-phage correlation: near 0.** Reason: the prediction for phage `j` is a single constant value across all test strains, so it cannot correlate with anything except by fold-to-fold noise.

**This asymmetry is exactly why the brief specifies both metrics.** Within-phage correlation is the honest test that your bacterial features are doing something -- if it is near zero, your model is the no-model baseline in disguise.

### 3.4 Realistic target numbers

Extrapolating from Gaborieau et al.'s binarized AUROC (Area Under the Receiver Operating Characteristic curve) of 0.86 to ordinal regression:

| Method | Within-host Pearson | Within-phage Pearson |
|---|---|---|
| No-model baseline (per-phage mean) | 0.30-0.45 | ~0.00 |
| Phylogeny only (ANI-MDS-32 or UMAP-8 of core distances) | 0.45-0.60 | 0.20-0.35 |
| Hand-crafted Gaborieau features | 0.55-0.70 | 0.40-0.55 |
| Hand-crafted + ESM-2 receptor embeddings (best realistic) | 0.60-0.75 | 0.45-0.60 |

These are educated-guess ranges, not promises. Report whatever you get against the same folds.

### 3.5 The overfitting cliff

With n = 402 effective samples (or ~300 after grouping near-clones), here is the dimensionality budget:

| Representation | Dimensionality d | n/d ratio | Safe? |
|---|---|---|---|
| Gaborieau hand-crafted (one-hot encoded) | ~50 | 8:1 | yes |
| 7-mer counts (k-mers of length 7) | $4^7 = 16{,}384$ | 0.02:1 | no |
| 8-mer counts | $4^8 = 65{,}536$ | 0.006:1 | no |
| Pan-genome presence/absence (Panaroo) | 15,000-30,000 | 0.02:1 | no |
| Whole-genome k-mer + PCA to 32 components | 32 | 12:1 | yes |
| Mean-pooled ESM-2 embeddings over all proteins | 1280 | 0.3:1 | no |
| ESM-2 over ~10 specific receptor proteins, pooled | ~480 | 0.8:1 | borderline |
| MASH/sourmash sketch + PCoA-32 | 32 | 12:1 | yes |

Anything with d > ~100 needs:
1. Heavy L1 or L2 regularization, AND/OR
2. Biology-targeted dimensionality reduction (e.g., extract only receptor proteins, not all proteins), AND/OR
3. Parameter sharing across the 96 phages (one shared encoder, a small linear head with 96 outputs).

Naively doing "high-dim features -> MLP per phage" is on the wrong side of the cliff.

---

## 4. What "extracting features from a genome" actually means

Before diving into specific features, here is the universal preprocessing step that every downstream tool builds on.

### 4.1 Gene calling -- the universal first step

A bacterial genome is a 5,000,000-character string. Most ML tools and all biology-targeted features operate not on the raw string but on the **list of predicted protein-coding genes**. The first thing you do with any genome is call genes.

**Tool: Prodigal** ("PROkaryotic DYnamic programming Gene-finding ALgorithm"). Wrapped by pyrodigal in Python.

**What it does mechanically**: scans the 5 Mb string in all six reading frames (three forward, three reverse), identifies open reading frames (ORFs) -- stretches starting with a start codon (typically ATG) and ending at a stop codon -- and scores each one with a Markov model trained on genuine coding sequence. ORFs scoring above threshold are reported as predicted genes.

**Input**: a FASTA file of contigs (the genome assembly).

**Output**:
- A GFF (General Feature Format) file listing the location, strand, and ID of each predicted gene.
- A FASTA file of nucleotide sequences (one per gene).
- A FASTA file of amino acid sequences (one per protein, obtained by translating the gene via the standard codon table).

**Time**: 5-30 seconds per genome on one CPU.

**Numbers**: an *E. coli* genome typically yields **4500-5500 predicted proteins**. For all 402 genomes: 1.8-2.2 million protein sequences total.

### 4.2 Annotation -- assigning function to the genes

Gene calling tells you *where* the genes are. Annotation tells you *what they are*. This step assigns putative function to each predicted protein.

**Tools: Bakta or Prokka**. These are wrappers that internally run Prodigal for gene-calling, then for each predicted protein run a sequence of database searches:

- HMMER (a profile-HMM search tool) against Pfam, TIGRFAM, NCBIfam (Hidden Markov Model databases of protein families).
- BLAST or DIAMOND against UniProtKB/Swiss-Prot (a curated protein database).
- Specialized DBs for tRNAs, rRNAs, CRISPR repeats.

Each predicted gene receives a name (e.g., `ompA`, `waaL`, `kpsM`) and a brief functional description. If no hit, it is labeled `hypothetical protein` (~10-25% of genes typically).

**Input**: a FASTA file of contigs.

**Output**:
- GFF, GenBank, and TSV files with one row per gene, including gene name, product description, and database cross-references.
- Amino acid FASTA of all proteins, with descriptive headers.

**Time**: Prokka ~4 min/genome on 8 CPUs; Bakta ~7 min/genome on 8 CPUs (more thorough). For 402 genomes on a 32-core machine: 30 minutes (Prokka) to 1.5 hours (Bakta).

**This step is the prerequisite for almost every "biology-targeted" feature.** Without annotation you do not know which protein is OmpA and which is a hypothetical of unknown function.

---

## 5. How the hand-crafted features are computed (mechanically)

This section walks through Gaborieau et al.'s feature pipeline step by step. These are the features you have to beat, so understanding exactly what each one captures is necessary.

### 5.1 O-antigen serotype -- via ECTyper

**What this feature is biologically**: The O-antigen is the outer polysaccharide of LPS, made by a cluster of 10-25 genes called the *rfb* cluster. There are >180 known *E. coli* O-types. Each type's gene cluster has a characteristic combination of glycosyltransferases (enzymes that add specific sugars in specific linkages).

**Tool: ECTyper** ("*E. coli* serotype Typer"), maintained by the Public Health Agency of Canada.

**What it does mechanically**:
1. Takes the assembled genome as input.
2. BLASTs (Basic Local Alignment Search Tool) the genome against a curated reference database containing one representative sequence per O-type's *wzx* and *wzy* genes (the two diagnostic genes of the O-antigen cluster).
3. Reports the best-matching O-type if hit identity and coverage thresholds are met; otherwise "O-untypeable" or "ONT" (not typed).

**Input**: FASTA file of the assembled genome.

**Output**: a single categorical label per strain, like `O157` or `O25` or `O-untypeable`. ECTyper also reports H-antigen (flagellar antigen) the same way, giving "serotypes" like `O157:H7`.

**Time**: 30 seconds per genome.

**Failure mode and ML implication**: ECTyper non-types about 9-25% of *E. coli* genomes in benchmarks (Bessonov et al. 2021), often because the strain has a novel or hybrid O-antigen cluster that does not match any reference. Those non-typed strains are exactly where learned representations can add value -- they encode the surface differently from any reference but still infect/resist phages in patterned ways.

**Featurization for ML**: one-hot encoding. If the panel has 93 distinct O-types, you get 93 binary columns (or 94 with "untypeable"). For rare types (one strain only), the column is essentially useless and may be dropped.

### 5.2 LPS outer-core type -- via WaaL phylogeny

**What this feature is biologically**: The outer core is the short sugar chain underneath the O-antigen. WaaL is the enzyme (O-antigen ligase) that attaches the O-antigen to the outer core; its sequence varies between strains and correlates with which outer-core structure the strain produces. *E. coli* has ~5 known outer-core types (called R1, R2, R3, R4, K-12).

**No off-the-shelf tool exists for this in Gaborieau's pipeline.** They built it as follows:

1. From each annotated genome (output of Sec.4.2), extract the WaaL protein sequence (annotation will label one protein `waaL` per genome).
2. Concatenate all 402 WaaL sequences into a single FASTA file.
3. Build a multiple sequence alignment (MSA) with MAFFT or muscle.
4. Build a phylogenetic tree of WaaL with IQ-TREE or FastTree.
5. **Manually inspect the tree and cut it into clades** corresponding to known outer-core types using literature references for type strains.
6. Assign each strain a categorical label for its outer-core clade.

**Input**: 402 annotated genomes.

**Output**: one categorical label per strain.

**Time**: build the alignment and tree in ~1 hour; manual clade-cutting is the actual bottleneck and is a one-time effort.

**ML implication**: this is a labor-intensive feature that captures real biology. You will not redo this work in 6 hours; you use the pre-computed labels from the paper's repository (it is in their GitHub).

**Featurization for ML**: one-hot, ~5-10 columns.

### 5.3 K-antigen capsule -- via CapsuleFinder and Kaptive

**What this feature is biologically**: Some strains have a polysaccharide capsule sitting on top of the LPS. Two kinds exist:
- **Group 1 capsules** (Klebsiella-style, *cps/wba* locus). Detected by Kaptive.
- **Group 2/3 capsules** (ABC-transporter-dependent, *kps* locus). Detected by CapsuleFinder.

**Tool 1: Kaptive** (Wyres/Holt, originally for *Klebsiella*).

**What it does**: BLASTs the genome against a curated database of known capsule loci, finds the best-matching locus and assigns a K-type. Same general approach as ECTyper but for the capsule gene cluster.

**Time**: 1-2 min per genome.

**Tool 2: CapsuleFinder** (built on **MacSyFinder** v2 -- "Macromolecular System Finder", a tool for detecting multi-gene systems with HMMs and gene-order constraints).

**What it does**: runs a set of HMMs (Hidden Markov Models) against your annotated proteome, checking for the characteristic combinations of *kps* genes that define group 2/3 capsule biosynthesis. Reports presence/absence of the system and a putative K-type if a serotype-specific signature is found.

**Time**: ~1 min per genome.

**Paper-specific numbers (Picard collection, n=403 per paper text)**: 22 strains (5.4%) encode a *Klebsiella*-style capsule (all in phylogroups A/B1/C, O-antigen types O8/O9/O89). 171 strains encode an ABC-dependent capsule; 101 of those were K-typed, dominated by K1 (n=42), K5 (n=19), K4 (n=11), K2 (n=10), K15 (n=6), K7 (n=6), K10 (n=1).

**Featurization for ML**: one-hot capsule type, plus a "no capsule" category (a substantial fraction of strains lack any detectable capsule). Probably ~20 columns total.

### 5.4 Sequence Type (ST) and phylogroup

**What these are biologically**: coarse phylogenetic labels.

- **ST** (Sequence Type) comes from **MLST** (Multi-Locus Sequence Typing). For *E. coli*, MLST uses seven housekeeping genes (e.g., *adk*, *fumC*, *gyrB*, *icd*, *mdh*, *purA*, *recA*). Each gene has many alleles; the combination of seven allele IDs defines an ST number. STs are reused across the literature for outbreak tracking.
- **Phylogroup** is a higher-level classification into eight clades (A, B1, B2, C, D, E, F, G), determined by which combination of marker genes is present.

**Tools**:
- **SRST2** ("Short Read Sequence Typing") or the older **mlst** tool by Torsten Seemann: BLAST the seven housekeeping genes against a reference allele database, look up the ST in the PubMLST database. ~10 seconds per genome.
- **ClermonTyping**: a quadriplex-PCR-in-silico method based on four marker genes. ~5 seconds per genome.

**Output**: one ST integer (or "novel"), one phylogroup label.

**Featurization for ML**: one-hot encoding of phylogroup (8-9 columns), and one-hot of ST (could be 100+ columns; usually drop rare STs or bin them).

### 5.5 Core-genome phylogeny -> UMAP-8

**What this is biologically**: a low-dimensional continuous embedding of the phylogenetic position of each strain. Captures relatedness beyond what discrete labels (ST, phylogroup) encode.

**Pipeline**:

1. Run **Panaroo** (Tonkin-Hill et al. 2020) on all 402 annotated genomes simultaneously. Panaroo clusters predicted proteins across genomes into orthogroups using a graph-based approach. It identifies the core genome -- orthogroups present in $\geq 99\%$ of strains. For 402 *E. coli* genomes, expect ~3000-4000 core orthogroups.

2. For each core orthogroup, take the gene sequence from each strain and align them (typically via MAFFT integrated into Panaroo).

3. Concatenate all core gene alignments into a single **core-genome alignment** of length 3-5 Mb (one row per strain, one column per nucleotide position).

4. Compute pairwise distances: for each pair of strains, count differing positions divided by total alignable positions. Result: a $402 \times 402$ distance matrix.

5. Reduce to 8 continuous dimensions using **UMAP** (Uniform Manifold Approximation and Projection -- a nonlinear dimensionality reduction). Note: Gaborieau used UMAP rather than the more conventional PCoA (Principal Coordinates Analysis) or MDS (MultiDimensional Scaling); UMAP is non-deterministic so they fixed a random seed.

**Input**: 402 annotated genomes.

**Output**: a $402 \times 8$ matrix of continuous features.

**Time**: Panaroo run is the bottleneck -- 1-3 hours on 16 CPUs with 32 GB RAM. The downstream alignment, distance, and UMAP take another hour.

**Featurization for ML**: 8 continuous columns. Already low-dimensional and well-conditioned.

### 5.6 Outer Membrane Proteins (OMPs) -- extracted but unused in the paper

Gaborieau et al. extracted OMP sequences and clustered them but did **not** include them in the final predictive feature set. This is an obvious gap.

**Procedure they used**: for each of the 12 OMP genes from Sec.2.1 (OmpA, OmpC, OmpF, LamB, FhuA, BtuB, NfrA, LptD, Tsx, TolC, YncD, FadL), extract the protein sequence from each annotated genome, cluster all 402 sequences per OMP using CD-HIT or MMseqs2 at a chosen identity threshold (e.g., 90%), assign each strain a cluster ID per OMP.

**About the shipped clustering data.** The directory is `data/genomics/bacteria/outer_membrane_proteins/` (no `_clustered` suffix), and it contains a single wide TSV `blast_results_cured_clusters=99_wide.tsv` with cluster IDs per strain x OMP at 99% identity -- **not** pre-extracted protein sequences. To get actual amino-acid sequences, run Bakta/Prokka on the FASTAs and join via locus tags.

**Featurization possibilities for ML**:
- Categorical cluster ID per OMP (~12 features, one-hot expanded).
- Mean-pooled ESM-2 embedding of each OMP sequence (~$12 \times 480 = 5760$ dims for the t12 model; PCA reduce).
- Pairwise OMP sequence-identity matrix to a reference per OMP (~12 continuous features).

This is the most accessible place to add value over Gaborieau's published feature set without major engineering.

### 5.7 Defense systems -- via DefenseFinder

**What this captures biologically**: per-strain inventory of antiphage defenses.

**Tool: DefenseFinder** (Tesson et al. 2022; Neron et al. 2023), built on MacSyFinder v2.

**What it does mechanically**: takes an annotated proteome, runs a curated library of ~150 HMMs against it, each HMM trained to detect proteins from a known defense system. Applies gene-order and co-occurrence constraints to confirm that the genes form a functional system, not just isolated homologs.

**Input**: annotated FASTA of all proteins (output of Sec.4.2).

**Output**: a TSV (tab-separated values) file listing each detected defense system, its subtype, and the genes participating in it.

**Time**: ~30 seconds per genome. For 402 genomes on 32 cores: ~10 minutes total wall-clock.

**Featurization for ML**: binary presence/absence vector. With ~150 system families, the vector is ~150-dimensional. In practice, many systems are rare; you may want to filter to systems present in $\geq 5$ strains (typically ~50-80 systems pass).

**Empirical caveat (from Gaborieau et al.)**: this feature contributed marginally to predictive performance in their binarized analysis. It is worth including but not worth engineering elaborately. **Hypothesis worth checking**: defense systems may matter more for the ordinal 0-4 score than for the binary lytic/non-lytic call, because they shift potency (4 -> 2) without necessarily abolishing lysis. This is a cheap empirical check to do during the onsite.

### 5.8 Concatenating everything

The full Gaborieau hand-crafted feature vector per strain:

| Feature group | Dim | Encoding |
|---|---|---|
| O-antigen | ~93 | one-hot |
| H-antigen | ~41 | one-hot |
| LPS outer core | ~5-10 | one-hot |
| K-type | ~20 | one-hot |
| ST | many; usually drop rare | one-hot or dropped |
| Phylogroup | 8 | one-hot |
| UMAP of core phylogeny | 8 | continuous |
| Total (after dropping rare categories) | ~50-80 | mixed |

This ~50-80-dimensional vector, fed to L1/L2-regularized logistic regression or a shallow random forest (one model per phage), is Gaborieau et al.'s entire predictor. They report **two** AUROC numbers from the same 10-fold grouped CV: **77% averaged across the 96 per-phage classifiers** (Fig 5A) and **86% on the aggregated prediction matrix** (Fig 5B). The 77% per-phage average is the more honest single-classifier metric; the 86% aggregated number reflects pooling across phages with very different positive-class prevalences.

---

## 6. Other representation options

This section catalogs alternative bacterial-genome representations beyond the hand-crafted ones, with mechanical descriptions and feasibility notes.

### 6.1 k-mer based representations

A k-mer is a substring of length k. For a genome of length L, there are L - k + 1 overlapping k-mers.

**Raw k-mer counts**:
- Tool: **jellyfish** or **KMC** (k-mer counting tools).
- For each genome and each chosen k, count occurrences of every length-k substring.
- For k = 7: $4^7 = 16{,}384$ possible k-mers; vector is 16,384-dimensional.
- Time: seconds per genome.
- Issue: dimensionality > n. Needs PCA, L1, or some other reduction.

**MinHash sketch (Mash, sourmash)**:
- Tool: **Mash** (`mash sketch -k 21 -s 10000 *.fasta`) or **sourmash**.
- Mechanism: for each genome, hash every k-mer (typically k = 21 or 31), keep the s smallest hash values (typically s = 1000-10000). This sketch is a randomized compressed representation that preserves Jaccard similarity between genomes.
- Output: per-genome list of s hash values, or pairwise distance matrix via `mash dist`.
- Time: seconds for sketching, ~10 seconds for $402 \times 402$ pairwise distance.
- Featurization: use the $402 \times 402$ distance matrix; apply PCoA or MDS to get 32 continuous components.

**Strong baseline**: sourmash sketch (k=31, scaled=1000) -> pairwise distance matrix -> PCoA-32. Captures phylogeny + accessory genome content implicitly. ~12 minutes total compute. Will likely match or beat phylogeny-only models but not the full hand-crafted set.

### 6.2 Pan-genome presence/absence

Already described in Sec.2.3 and Sec.5.5.

**Tool: Panaroo** (preferred over the older Roary). Outputs `gene_presence_absence.csv` -- a $(402 \times {\sim}20{,}000)$ binary matrix.

**Time**: 1-3 hours on 16 CPUs for 402 genomes (this includes its own annotation; can be sped up if you pre-annotate with Bakta).

**Featurization options**:
- **All genes** (~20,000 columns): catastrophic overfit with n=402 unless heavily regularized. Lasso/elastic net at very high $\alpha$ might rescue it, but not recommended.
- **Filtered to surface-related genes**: keep only orthogroups whose annotation matches `waaL`, `wzx`, `wzy`, `rfb*`, `wb*`, `kps*`, `ompA`, `ompC`, `ompF`, `lamB`, `fhuA`, `btuB`, `fepA`, `lptD`, `tolC`, `tsx`, `fim*`, `pap*`, `sfa*`, `foc*`. Probably ~100-300 columns. **This is high-EV.**
- **PCA of full matrix**: project $(402 \times 20{,}000)$ to $(402 \times 50)$. Loses interpretability.

### 6.3 Protein language model embeddings -- ESM-2

**ESM-2** ("Evolutionary Scale Modeling 2", Lin et al. 2023) is a Transformer trained on ~65 million protein sequences from UniRef50 with a masked-residue objective. It outputs a per-residue embedding for any input protein sequence. Available sizes from 8M to 15B parameters; the 35M-parameter `esm2_t12_35M_UR50D` is a reasonable speed-quality tradeoff for this project.

**Mechanically, how to use it**:
1. Take the annotated proteome of a genome (output of Sec.4.2).
2. **Curate a list of receptor-related proteins** (the 12 OMPs from Sec.2.1: OmpA, OmpC, OmpF, LamB, FhuA, BtuB, NfrA, LptD, Tsx, TolC, YncD, FadL -- plus the surface-polysaccharide enzymes WaaL and key Wb*, Kps* enzymes if you want). About 13-18 proteins per genome.
3. For each of these proteins, run ESM-2 forward pass (1 protein at a time on a GPU; t12 model fits comfortably on a T4 or better).
4. ESM-2 outputs a tensor of shape (L, 480) where L is the protein length and 480 is the t12 embedding dimension. **Mean-pool over the L axis** to get a single 480-dim vector per protein.
5. Concatenate across the curated proteins: ~$15 \times 480 = 7200$ dims per genome. PCA to ~64 components.

**Time**: ~1-2 GPU-hours for all 402 genomes if you curate to ~15 proteins each. Without curation (all 5000 proteins per genome): >20 GPU-hours and not feasible in the onsite.

**Why this is plausibly the best learned representation to add**: it captures intra-serotype sequence variation in the actual receptor proteins. ECTyper would call two strains both "O157" even if their *waaL* differs by 20 amino acids in a region that affects phage binding; ESM-2 captures that difference.

### 6.4 Genomic language models -- not realistic in 6 hours

**Evo, Evo 2** (Arc Institute) -- Transformers on raw nucleotide sequence with up to 1 Mb context. The 1B-parameter model fits on an A100; the 40B requires multiple H100s. Inference is slow and you would need to chunk the 5 Mb genome.

**Nucleotide Transformer** (InstaDeep) -- up to 2.5B params, ~6 kb context. Same chunking issue.

**DNABERT-2** -- BPE-tokenized, more efficient per token, but still <10 kb effective context.

**gLM, gLM2** (Hwang lab; Tatta Bio) -- operate on protein tokens rather than raw DNA, more suited to bacterial-genome scales but still per-gene.

For 6 hours, the only realistic plan involving these is: extract specific receptor genes and embed them via a genomic LM at gene scale. ESM-2 over proteins is the same idea with mature, fast, well-documented tooling, and is recommended over genomic LMs unless you have strong prior reason.

### 6.5 ANI-based representation

**Tool: skani** (Shaw & Yu 2023). Fastest all-vs-all ANI tool currently available.

**What it does**: computes Average Nucleotide Identity between every pair of genomes using sparse k-mer chaining.

**Time**: ~1 minute for $402 \times 402$ on a laptop.

**Output**: a $402 \times 402$ distance matrix.

**Featurization**: MDS or PCoA to 32 continuous components. Captures phylogeny but not accessory variation. Use as a baseline and for the CV grouping in Sec.3.1; not strong as a standalone predictive representation.

---

## 7. Reference papers and software

Numbers in parentheses are publication years.

**The dataset paper**:
- Gaborieau et al. (2024). "Prediction of strain level phage-host interactions across the Escherichia genus using only genomic information." *Nature Microbiology* 9:2847-2861. DOI 10.1038/s41564-024-01832-5. Code: github.com/mdmparis/coli_phage_interactions_2023. Data on Zenodo (10.5281/zenodo.10202713) and figshare (10.6084/m9.figshare.25941691.v1).

**Closest methodological cousin**:
- Boeckaerts et al. (2024). "Predicting Klebsiella phage host specificity at the strain level." *Nature Communications* 15:4355. DOI 10.1038/s41467-024-48675-6. ESM-2 embeddings of phage RBPs and bacterial K-loci -> XGBoost. Repo: github.com/dimiboeckaerts/PhageHostLearn.

**Genome annotation**:
- Hyatt et al. (2010). "Prodigal: prokaryotic gene recognition." *BMC Bioinformatics* 11:119.
- Seemann (2014). "Prokka: rapid prokaryotic genome annotation." *Bioinformatics* 30:2068.
- Schwengers et al. (2021). "Bakta: rapid and standardized annotation of bacterial genomes via alignment-free sequence identification." *Microbial Genomics* 7:000685.

**Serotyping**:
- Bessonov et al. (2021). "ECTyper: in silico Escherichia coli serotype and species prediction." *Microbial Genomics* 7:000728.
- Wyres et al. (2016). "Identification of Klebsiella capsule synthesis loci from whole genome data" [Kaptive]. *Microbial Genomics* 2:e000102.

**Pan-genome and phylogeny**:
- Tonkin-Hill et al. (2020). "Producing polished prokaryotic pangenomes with the Panaroo pipeline." *Genome Biology* 21:180.
- Shaw & Yu (2023). "Fast and robust metagenomic sequence comparison through sparse chaining with skani." *Nature Methods* 20:1661.
- Ondov et al. (2016). "Mash: fast genome and metagenome distance estimation using MinHash." *Genome Biology* 17:132.

**Defense systems**:
- Tesson et al. (2022). "Systematic and quantitative view of the antiviral arsenal of prokaryotes." *Nature Communications* 13:2561 [DefenseFinder].
- Payne et al. (2021). "PADLOC: a web server for the identification of antiviral defence systems." *NAR* 49:W279.

**Protein language models**:
- Lin et al. (2023). "Evolutionary-scale prediction of atomic-level protein structure" [ESM-2]. *Science* 379:1123.

**Phage-host prediction tools** (most are phage-side; included for context):
- PHIST (Zielezinski et al. 2022): k-mer matches. github.com/refresh-bio/PHIST.
- iPHoP (Roux et al. 2023): ensemble. bitbucket.org/srouxjgi/iphop.
- CHERRY (Shang & Sun 2022): graph CN. github.com/KennthShang/CHERRY.
- vHULK (Amgarten et al. 2022): NN on phage protein families.
- WIsH (Galiez 2017): Markov models.

---

## 8. Hour-by-hour plan for the 6-hour onsite

Numbers in brackets denote the cumulative wall-clock hour.

**[0.0-0.5h] Setup and eval harness.**

- Inspect the data: confirm matrix shape (402, 96), distribution of values, missingness, FASTA file naming consistency.
- Ask Tabula: what are the pre-computed representations they mentioned? Are the hand-crafted Gaborieau features already provided as a CSV?
- Write the evaluation harness:
  - 10-fold GroupKFold with groups computed from skani ANI $\geq 99.99\%$.
  - Functions for the four metrics (within-host x within-phage x Spearman x Pearson).
  - No-model baseline (per-phage mean).
  - Per-host mean baseline.
- Verify that the no-model baseline reproduces the expected pattern: within-host Pearson positive (~0.3-0.5), within-phage Pearson near 0.
- Deliverable: a `Featurizer` abstract class with `.fit(genome_list)` and `.transform(genome) -> 1d np.array`.

**[0.5-2.0h] Reproduce the hand-crafted Gaborieau baseline.**

- If they provided the features as a CSV: feed directly into per-phage Ridge or shallow Random Forest. Validate that you get within-phage Pearson $\approx 0.4\text{-}0.55$.
- If not: at minimum run ECTyper (~10 min for 402 genomes), ClermonTyping (~5 min), skani-based phylogeny PCoA-8 (~5 min). Skip the more elaborate features (WaaL phylogeny, Capsule, Defense systems) unless time permits.
- Concatenate and train; report the four metrics.
- Deliverable: a number for the hand-crafted reference on your CV folds.

**[2.0-4.0h] Two learned representations.**

- (a) **sourmash sketch + PCoA-32** -- implemented in background; ~15 min total. This is your cheap-but-strong learned baseline. Compare against hand-crafted.
- (b) **Targeted Panaroo on surface gene families** -- Panaroo on Bakta-annotated genomes (kick off early; takes ~2h in background), then filter to ~200 surface-related orthogroups. Concatenate to hand-crafted; check if it adds.

**[4.0-5.5h] ESM-2 receptor embeddings (the high-EV experiment).**

- From annotated genomes, extract the 12 receptor OMPs per strain plus WaaL.
- Mean-pool ESM-2 t12 (35M) embeddings -> ~5300 dims per genome.
- PCA to 64. Concatenate to hand-crafted. Train Ridge. Compare.
- If you have time: train a joint multi-output model with shared 64-dim encoder and 96-head linear output, using ordinal-aware MSE loss.

**[5.5-6.0h] Write up.**

- Single table comparing 4-6 representations across 4 metric variants. Each row is a featurizer, each column is a metric.
- One sentence per row: what does this feature capture biologically, did it help or not, why.
- Honest list of what you would try next and what you would not.

---

## 9. Failure modes and pitfalls

**Phylogenetic data leakage.** Covered in Sec.3.1. The single biggest threat to honest evaluation.

**Within-host vs within-phage tradeoff.** A model that predicts the per-host mean perfectly but ignores phage identity achieves great within-host correlation and near-zero within-phage. Always report both.

**Per-phage class imbalance.** Some phages have $<5$ lytic interactions across 402 strains. Per-phage correlations for such phages are statistically meaningless. Report n_lytic alongside per-phage metrics; consider thresholding to phages with $\geq 10$ lytic strains for the "stable" portion of the analysis.

**One-hot encoding of rare categories.** O-type with one occurrence becomes a column that perfectly predicts one strain. Drop columns appearing in <3 strains or use leave-one-out target encoding.

**Mismatched normalization between train and test.** Apply any standardization on the training fold's statistics only, then transform the test fold. Not a Pearson-correlation issue (Pearson is scale-invariant) but matters for Spearman ties and for downstream losses.

**Ordinal vs binary mismatch with Gaborieau.** Gaborieau et al. binarized to lytic/non-lytic. Your task is ordinal regression on 0-4. Their AUROC numbers translate only roughly to your Pearson/Spearman; do not promise to "match the paper" -- instead, run their feature set on your ordinal task with your eval harness, and report that as the calibrated reference.

**Annotation errors propagating silently.** If Bakta mislabels a paralog as `ompA`, ESM-2 will embed the wrong protein. Always sanity-check the extracted receptor sequences (length, HMM hit score) before embedding.

**The interviewers will likely ask: how does this generalize to a new phage?** Honest answer: it does not, by construction. The phages are fixed output dimensions. To generalize to new phages you would need to featurize phages (their RBPs via ESM-2 is the standard approach, as in Boeckaerts 2024) and reformulate as pairwise prediction.

---

## 10. What Tabula is actually evaluating

Re-reading the brief carefully, in order of explicit emphasis:

1. **Reasoning.** "We are especially interested in your reasoning: what you choose to try, why you think it might work, how you evaluate it, and how you interpret the results."
2. **Representations.** "The goal is to explore host genome representations" and "The main focus is the representation, not extensive tuning of the predictive head."
3. **Comparison.** Against both the no-model baseline and the hand-crafted reference on the same folds.
4. **Honesty.** "An honest read on what seems to be working, what is not, and what we might try next."
5. **Clean code.** The `featurizer.fit_transform(genomes)` API hint suggests they want swappable featurizers.
6. **Use of coding agents.** "We suggest you use a coding agent extensively!" -- they want to see effective delegation.

Things they explicitly do **not** want:
- Polished pipeline ("We do not expect a polished pipeline").
- Final answer.
- Extensive hyperparameter tuning.
- Beating the Nature Microbiology paper (although it would be nice).

The deliverable is a coherent story: I picked X because Y; here is what I observed; here is what I would do next. The numbers serve the story, not the other way around.
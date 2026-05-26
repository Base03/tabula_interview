# Onsite Project: Exploring Host Genome Representations for Phage-Host Prediction

## Overview

This is a **~6 hour onsite project** focused on an active research problem at Tabula: how should we represent bacterial host genomes in a way that is useful for predicting phage tropism?

We do not expect a polished pipeline or a final answer. The goal is to work together, think through what information in a bacterial genome might matter for phage susceptibility, test ideas, and see what the data tells us.

We are especially interested in your reasoning: what you choose to try, why you think it might work, how you evaluate it, and how you interpret the results.

## Background

For this project, we will use the dataset from Gaborieau et al. 2024 (_Nature Microbiology_), which experimentally measured interactions between a diverse collection of _Escherichia_ strains and a panel of virulent phages.

This is a particularly useful dataset for us because it gives a concrete way to test ideas about host representation. Given a bacterial genome, can we build a representation that helps predict which phages will infect it?

The original study used **hand-crafted bacterial features**, including:

- surface polysaccharide serotypes: O-antigen, LPS outer core, K-antigen capsule
- outer membrane protein variants
- antiphage defense system inventories
- a low-dimensional embedding of the core-genome phylogeny

Those features are a strong starting point. In this project, we want to explore whether other genome representations -- learned, sequence-derived, biologically motivated, or simpler baselines -- can capture useful signal as well.

We will frame the task as **multi-output regression**: the model takes one bacterial genome as input and predicts a vector of phage interaction scores (one for each phage in the dataset). Note that phages are not featurized (they are represented implicitly as different regression targets). Comparing performance under this regression setup is one way to evaluate how much useful signal different bacterial genome representations contain.

## Data

You will receive:

- **402 bacterial genomes** as assembled FASTA files, spanning the _Escherichia_ genus.
- An **interaction matrix** of shape `(402 x 96)` with interaction scores in `{0, 1, 2, 3, 4}`, where `0` means no observed lytic interaction and higher values indicate stronger lysis.
- The paper's **hand-crafted bacterial features**, which we can use as a reference featurization.
- Some additional **pre-computed genome representations** that we have been experimenting with internally. These are available if useful.

Phages are treated as fixed output dimensions. The model takes a single bacterial genome as input and predicts the full vector of 96 interaction scores.

## Project Goal

The goal is to explore host genome representations.

A representation could be simple or complex. For example, it might be based on k-mers, genes, protein families, defense systems, surface-associated genes, phylogeny, learned embeddings, or some combination of these. We are not looking for one specific approach.

A good outcome would be:

- a clear idea for one or more host representations
- a simple pipeline for converting genomes into fixed-dimensional features
- a predictive model trained on those features
- comparison against a baseline and the provided hand-crafted features
- an honest read on what seems to be working, what is not, and what we might try next

The predictive model itself can be simple: linear regression, ridge regression, random forest, gradient boosting, a small MLP, or whatever seems reasonable. The main focus is the representation, not extensive tuning of the predictive head.

## Evaluation

1. **Splits.** 10-fold random cross-validation over the 402 strains. All 96 interaction scores for a held-out strain are predicted from a model trained on the other ~362 strains.

2. **Primary metrics.** Spearman and Pearson correlation between predicted and observed interaction scores, computed two ways:

    - **Within-host:** for each held-out strain, correlate its predicted 96-vector against its observed 96-vector. Average across held-out strains.
    - **Within-phage:** for each phage, correlate its predicted scores against observed scores across held-out strains. Average across phages.

    These measure complementary things. Within-host correlation tells you whether the model can rank phages correctly for a given strain (the relevant question for cocktail design). Within-phage correlation tells you whether the model can rank strains correctly for a given phage (the relevant question for host-range prediction).

1. **No-model baseline.** Predict each held-out host's 96-vector as the per-phage mean interaction score computed across the training hosts in that fold. This is the "no model" floor: the prediction ignores the held-out host's genome entirely. Within-host correlation under this baseline is non-trivial. It captures the global "average phage prevalence" pattern and is a meaningful floor to beat. Within-phage correlation should sit close to zero (the prediction for a phage only varies across hosts via small fold-to-fold differences in the training mean), so any featurization that actually uses the genome should clearly beat it on that axis.

2. **Hand-crafted reference.** Run the same model class you use for your own featurizations on the provided hand-crafted features.

## Suggested Workflow

We suggest you use a coding agent extensively! You may even want to try a swarm of agents. Welcome to the future :)
- Get a regression and eval framework setup on the server. You should be able to compare the hand-crafted reference to the no-model baseline.
- Research alternative representations. Limit this to 45 minutes, but certainly use LLMs (deep research). We want to see what you come up with if you research this on your own.
- Talk through our existing results. We'll talk about what we'e already tried (and not tried).
- Test! ML is an empirical science. Deploy the agent swarm!


It would be helpful to structure the code so that different representations can be swapped in easily. For example:
```python
X = featurizer.fit_transform(genomes)
Y = interaction_matrix
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
```

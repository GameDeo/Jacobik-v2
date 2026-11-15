# NCI v2 surface model

## Scope and implementation state

NCI v2 is a research model for transferable, geometry-dependent noncovalent interaction contributions. Its implementation is in `research/archive/code/interactions/nci_v2/nci_v2.py`; cohort construction and registry logic are in `research/archive/code/refit/nci_v2_frame.py` and `research/archive/code/refit/nci_v2_registry.py`. It is distinct from the older histogram-based NCI term in `research/archive/code/nci/nci.py` and is not selected by the production scorer. The code provides opportunity enumeration, differentiable features, surface fitting, calibration artifacts, and runtime evaluation interfaces. Adoption requires independent validation and explicit integration with the solver profile and database contract.

## Interaction families

An opportunity is generated from molecular graph and functional-group rules, then mapped to a continuous feature vector. The current schema defines three families:

| Family | Geometric anchor | Feature content |
|---|---|---|
| `bond_atom` | Directed donor or activated bond against an acceptor atom | Anchor-to-acceptor distance and orientation descriptors such as donor direction, acceptor lone-pair proxy, or sigma-hole direction. |
| `bond_ring` | A bond-centered donor or acceptor relative to an aromatic ring | Bond-to-ring-centroid distance plus orientation relative to the ring plane and bond direction. |
| `ring_ring` | A pair of aromatic rings | Centroid separation, relative plane orientation, and lateral displacement within the ring frame. |

Candidate rules are declarative and include polar-H/acceptor, activated C–H/acceptor, sigma-hole/acceptor, polar multiple bond, bond–ring, lone-pair or halogen–ring, and ring–ring opportunities. A candidate is a geometric opportunity, not a claim that an interaction is energetically favorable.

## Smooth geometric representation

All runtime features must remain differentiable under optimization, including coincident or nearly degenerate geometries. A displacement vector \(\mathbf v\) uses

\[
\lVert\mathbf v\rVert_\delta=\sqrt{\mathbf v\cdot\mathbf v+\delta_r^2},
\qquad
\widehat{\mathbf v}_\delta=\frac{\mathbf v}{\lVert\mathbf v\rVert_\delta}.
\]

For a ring with ordered vertices \(\mathbf p_i\), the centroid and oriented area vector are

\[
\mathbf c_R=\frac{1}{n}\sum_i\mathbf p_i,
\qquad
\mathbf s_R=\sum_i(\mathbf p_i-\mathbf c_R)\times(\mathbf p_{i+1}-\mathbf c_R).
\]

The regularized ring normal is \(\widehat{\mathbf n}_R=\mathbf s_R/\sqrt{\mathbf s_R\cdot\mathbf s_R+\delta_n^2}\). Ring identity is canonicalized over cyclic shifts and reversal; unique eligible 5- and 6-membered rings are derived from the ligand graph.

Feature bounds, support windows, and coordinate regularizers are part of a versioned surface specification. Inner and outer radial windows use quintic step functions whose values and first two derivatives join continuously. The same smooth cutoff convention applies to all opportunity families.

## Matched reference and background distributions

The statistical target is a density ratio between reference interaction geometries and an appropriate background distribution:

\[
F(\mathbf x,\mathbf z)\approx
\log\frac{p_{\mathrm{ref}}(\mathbf x\mid\mathbf z)}
               {p_{\mathrm{base}}(\mathbf x\mid\mathbf z)}.
\]

Here \(\mathbf x\) contains continuous geometry, and \(\mathbf z\) contains discrete chemical context. Reference opportunities come from structural data. Background opportunities are generated under matched chemistry and topology so that candidate abundance and accessible geometry are represented explicitly. Grouped splitting prevents related structures from appearing on both sides of a fit/evaluation boundary. Candidate multiplicity is retained so that runtime summation represents the same opportunity process used during fitting.

## Surface parameterization and support

The latent log-density ratio is an additive radial-plus-angular spline surface:

\[
F(r,\mathbf z)=f_r(r)+\sum_j f_{rj}(r,z_j).
\]

Each function is represented by a tensor product of clamped cubic B-spline bases. Cubic bases are (C^2) at simple interior knots and are evaluated in PyTorch so coordinate gradients flow through the surface. A regularized logistic fit separates reference opportunities from background opportunities. The objective includes second-difference roughness penalties and a ridge term to control unsupported oscillation and coefficient magnitude.

The fitted surface is accompanied by a joint-support function \(m(\mathbf x)\in[0,1]\). Support is assessed over independent structure groups and effective group mass, separately for reference and background observations. A surface contribution is active only where its reference and background support satisfy the declared rules. Reference-only support is tracked explicitly and requires calibration-motion traversal checks; it is not silently treated as matched density evidence.

## Positivity, tapering, and runtime score

To prevent the density-ratio estimate from turning undersampled regions into artificial attractions, the latent surface is mapped through a smooth positive-part function:

\[
h_\epsilon(v)=
\begin{cases}
0,&v\le 0,\\
\epsilon(6t^3-8t^4+3t^5),&0<v<\epsilon,\\
v,&v\ge\epsilon,
\end{cases}
\qquad t=v/\epsilon.
\]

The raw opportunity contribution combines the positive calibrated log-density ratio, support, and radial tapers. Each opportunity is evaluated from the current coordinates, and opportunity terms are aggregated according to the selected calibration artifact. Family scales and a shared scale are nonnegative and stored with the calibrated surface artifact. Their values are fitted offline, not chosen dynamically in the solve.

## Offline response calibration

The structural density ratio establishes where geometries are enriched in the reference structures; it does not establish the energy scale. NCI v2 therefore treats energy and force response as calibration labels. For labeled coordinate motions, the implementation compares surface-induced score changes and gradients with reference energy and force changes, then estimates nonnegative shared and family response scales under regularization. It reports a singular-value and condition-number diagnostic to expose poorly identifiable scale directions. This calibration is separate from surface fitting and from solver optimization.

## Artifact and runtime interfaces

The module defines typed artifacts for opportunity batches, feature batches, density-ratio surfaces, calibrated surfaces, and pre-indexed runtime evaluators. An artifact carries family identity, context surfaces, spline specifications, support masks, fitted coefficients, scales, schema metadata, and provenance. Runtime construction maps atom indices to solver coordinate slots and builds family-specific tensors once; evaluation then operates on candidate coordinate batches with autodiff.

`research/archive/code/refit/nci_v2_frame.py` constructs a candidate cohort from cleaned structures and audits graph-defined opportunity counts. `research/archive/code/refit/nci_v2_registry.py` validates experiment records and artifact provenance, including fingerprints and dependencies. These support experimental accounting; they do not themselves enable the component in `ComplexBuilder`.

## Scientific limits

The model depends on the candidate grammar, the match between reference and background opportunity sampling, feature support, and the identifiability of response calibration. Graph admission does not capture every chemical interaction. A smooth surface can still be biased by sparse chemistry, correlated opportunities, or a background generator that fails to represent accessible noninteracting geometries. The family definitions and calibration scheme therefore remain research interfaces until tested across held-out chemical contexts and full-complex solver motions.

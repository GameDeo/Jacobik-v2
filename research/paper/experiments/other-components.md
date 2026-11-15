# Other experimental components

## Legacy histogram NCI

The older NCI module (`research/archive/code/interactions/nci.py`) estimates statistical enrichment for selected noncovalent geometry classes from structural histograms. Its categories are:

- `hb`: polar-hydrogen to acceptor distance and donor–hydrogen–acceptor angle;
- `pp`: aromatic ring-pair centroid distance and plane orientation;
- `chp`: nonpolar hydrogen relative to an aromatic ring centroid and normal.

The fitted term is a tapered log ratio of observed crystal density to a random-placement reference density. Candidate selection uses ligand topology and excludes covalent neighborhoods. A shared candidate generator is needed across extraction, census, and scoring so that donor/acceptor exclusions have the same meaning in each stage. When the H-bond type is enabled, its polar-H attraction replaces the corresponding Q6 contact well. This model is separate from NCI v2 and has its own histogram and database schema.

## Bond-valence coupling

The optional QV component (`research/archive/code/interactions/bvs.py`) supplements independent Q1 bond-distance terms with a complex-level constraint on their shared valence sum. For sigma bonds, the bond-valence contribution is

\[
s_i=\exp\!\left(\frac{R_{0,i}-R_i}{B}\right),\qquad
V=\sum_i s_i.
\]

Reference \(R_0\) parameters are fitted through metal/donor and periodic-group fallback levels. At state-conditioned levels, the dimensionless target is \(x=V/\mathrm{OS}\); at a metal-only level, the reference scale is based on the median valence sum. The runtime reference-ratio term is quadratic:

\[
\Delta\log p(x)=-\frac{c}{2}(x-\mu)^2+\mathrm{constant},
\qquad
c=\max\!\left(0,\frac{1}{s_{\mathrm{obs}}^2}-\frac{1}{s_{\mathrm{indep}}^2}\right).
\]

The independent spread is estimated by shuffling bond lengths within the corresponding Q1 keys. This leaves only the extra covariance among bonds beyond what Q1 already scores. The current implementation is scoped to all-sigma coordination environments and excludes haptic complexes. Its profile field is disabled by default; table availability alone does not activate it.

## Metal 1–3 contact model

The optional `q6_m13` path represents a metal-to-atom contact where the nonmetal atom is covalently adjacent to a donor atom. It has a separate topology class for ordinary and haptic donor attachment and can use fitted wall or short-range core entries. This is distinct from direct metal–donor coordination (owned by Q1) and from general class-5 nonbonded contacts. Its experimental status is separate from the adopted Q6 model.

## Search coverage (S3)

The current torsion beam ranks partial ligand seeds from Q5 modes without seeing the metal or other ligands. Search-coverage work explores coordination-aware beam expansion, improved assignment starts, and basin diversity. The model contract is to change candidate generation while preserving the scorer and profile used to rank a fixed geometry. Search changes can affect the final answer through coverage even when re-ranking an unchanged pool is identical.

## Ligand internal geometry (S4)

The default kinematic representation freezes most bond lengths and bond angles at the free-ligand embed, with optional localized bends and explicit torsion or ring coordinates. Internal-geometry research covers donor-bound ligand distortion, discrete stereochemical alternatives, small-ring pucker alternatives, and macrocycle conformational coverage. Any additional coordinate needs an associated prior or a principled enumeration rule; otherwise the search introduces unconstrained degrees of freedom that the existing feature score may not identify.

## Bond-valence revisions and dependencies

Possible refinements to QV include conditioning reference distances and valence levels on coordination environment and spin, and separating pi-acceptor carbon donors from ordinary sigma-bond behavior. These are model-design directions, not part of the current QV runtime contract. Such changes require new fitted tables and an explicit profile selection. Their calibration must preserve the meaning of Q1 and avoid counting an existing distance preference twice.

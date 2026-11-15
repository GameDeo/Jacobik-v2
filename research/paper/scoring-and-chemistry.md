# Scoring and chemical models

## Score definition

For a complex with feature set \(\mathcal F\), the adopted summed objective is

\[
S(\mathbf X)=\frac{1}{Z}\left[
\sum_{f\in\mathcal F}\log p_f\!\left(x_f(\mathbf X)\mid k_f\right)
+ S_{\mathrm{phys}}(\mathbf X)+S_{\mathrm{bend}}(\mathbf X)
\right],
\]

where \(\mathbf X\) denotes candidate coordinates, \(x_f\) is a feature, \(k_f\) its categorical key, and \(Z\) is a fixed per-complex normalization. In the implementation, physical contact and bend contributions are expressed on the same score scale and normalized consistently with the active profile. Larger scores are preferred. The score is a sum of log densities and physical exclusion terms; it is not an energy and does not define a probability distribution over complete conformers without an additional calibrated statistical-mechanical model.

The current objective combines metal–donor distances (Q1), donor–metal–donor angles (Q2), joint donor orientation (QO / \(S_{\mathrm{orient}}\)), torsions (Q5), nonbonded contacts (Q6), and enabled bend priors. Bond-valence coupling (QV) and noncovalent interaction surfaces are separate optional components. Shape and symmetry penalties enforce solver constraints and are not feature likelihoods.

For a univariate finite mixture with component densities \(f_j\),

\[
\log p(x\mid k)=\operatorname{LSE}_{j=1}^{K}\left(\log\pi_j+\log f_j(x)\right),\qquad
\operatorname{LSE}(a_1,\ldots,a_K)=\log\sum_j e^{a_j}.
\]

This log-sum-exp marginalizes the latent component identity. The runtime applies variance floors before evaluating narrow components. The `u1` profile uses the summed log-likelihood over eligible features divided by \(Z\); it does not average each feature family independently.

## Fitted feature densities

### Metal–donor distances: Q1

For donor group \(s\),

\[
d_s=\lVert \mathbf D_s-\mathbf M\rVert_2.
\]

The donor point is an atom for a kappa donor and a centroid for a haptic donor group. Q1 is a Gaussian mixture in distance. Its key can include metal, spin, oxidation state, donor subclass, coordination geometry, and trans-partner subclass. An ordered fallback ladder backs off to broader chemical groups when a specific table is unavailable. A trans partner is derived from the candidate assignment, so candidate assignments can use different Q1 keys.

### Coordination angles: Q2

For every pair of sites \(s,t\), Q2 scores

\[
c_{st}=\cos\angle(\mathbf D_s-\mathbf M,\mathbf D_t-\mathbf M).
\]

The fitted mixture is Student-t in cosine space. Keys encode geometry, the topological relationship of the donor groups, metal and donor classes, and the site-pair class where present. Cosine-space fitting avoids a separate angular-coordinate singularity in the distribution model.

### Joint donor orientation: QO

For a kappa donor with substituent unit vectors \(\mathbf u_i\) and metal direction \(\mathbf u_M\), the local orientation feature contains pairwise dot products, including \(\mathbf u_M\cdot\mathbf u_i\) and \(\mathbf u_i\cdot\mathbf u_j\). These are entries of a Gram representation of local bond geometry. A full-covariance Gaussian mixture captures correlations among the angles and replaces separate Q3 approach-angle and Q4 pyramidalization terms when QO data are available.

For haptic groups, features describe the orientation of the participating face relative to the metal and the face substituents. Haptic orientation is scored per substituent when the database supplies that representation.

### Torsions: Q5

Each eligible rotatable bond is described by a signed dihedral defined by the bond and its flanking atoms. The density is a von Mises mixture,

\[
p(\phi)=\sum_{k=1}^{K}\pi_k\frac{\exp\{\kappa_k\cos(\phi-\mu_k)\}}{2\pi I_0(\kappa_k)}.
\]

The key encodes endpoint and flank atom classes and ring context. Symmetry-equivalent rotor states are folded during extraction and handled consistently at scoring time. The Q5 term covers torsional preferences and, through training correlations, some 1–5 steric structure; a separate physical class-4 repulsion is not part of the adopted model.

## Contact chemistry: Q6

Contact candidates are generated from molecular topology with covalent-neighborhood exclusions. Contact handling distinguishes metal pairs, class-4 pairs, class-5 pairs, polar-hydrogen contacts, and metal 1–3 contacts. The model assigns short-range exclusion energy only in the supported domain, avoiding a fitted attractive PMF over all atom pairs.

For class-5 nonpolar pairs, the physical repulsion is a shifted exponential with zero value and slope at a van der Waals contact distance \(r_m\):

\[
U_{ij}(r)=\frac{A_{ij}}{RT}\left[e^{-b_{ij}r}-e^{-b_{ij}r_m}
 + b_{ij}e^{-b_{ij}r_m}(r-r_m)\right],\quad r<r_m,
\]

and \(U_{ij}(r)=0\) for \(r\ge r_m\). The score contribution is \(-U_{ij}\), summed over eligible pairs. Element parameters are fit offline to GFN2-xTB dimer interaction scans with a damped dispersion component included in the calibration model; the runtime term retains the repulsive component. A linear continuation at very short distances prevents numerical overflow and preserves a finite gradient.

Other pair classes use a statistical onset model: a data-derived onset distance \(r_0\) and an exponential core below onset. The contribution is zero at and beyond onset. Polar-hydrogen attraction is handled by the enabled H-bond component when that component is active; it is removed from the corresponding Q6 well to avoid duplicate scoring. Metal coordination distances and angles are owned by Q1, Q2, and QO; Q6 supplies only short-range exclusion for those relationships.

## Bend prior

An enabled bend coordinate \(\beta\) has a flat region around its embedded value followed by a harmonic penalty:

\[
E_{\mathrm{bend}}(\beta)=\frac{k}{2}\left[\max\left(0,|\beta-\beta_0|-\beta_{\mathrm{free}}\right)\right]^2.
\]

The stiffness class depends on hybridization and bend direction. These coefficients are calibrated offline against MACE single-point energies for isolated ligand distortions. The runtime uses the resulting analytic prior and does not call MACE.

## Thermal width floor

A fitted mixture may contain components narrower than the thermal motion of a molecule around one minimum. For a feature \(q(\mathbf x)\), the harmonic approximation gives

\[
\operatorname{Var}_T(q)\simeq k_BT\,\nabla q^{\mathsf T}H^+\nabla q,
\]

where \(H\) is the reference Cartesian Hessian after rigid modes are removed. Measured thermal widths are stored as model constants and used to floor fitted variances. For angular features represented as cosines, the angular width is propagated to cosine space at each component mean. For QO covariance matrices, the floor is applied in the positive-semidefinite ordering. No Hessian is computed during a solve.

## Charge and coordination chemistry

The solver estimates metal oxidation state from total complex charge and ligand ionic charges. The training-convention model applies donor-aware treatment for neutral hydrides and nitro ligands, counts each ligand's formal charge once, and does not infer a charge from target coordinates. Anionic donor chemistry must be represented explicitly in the ligand input. Unknown or implausible oxidation states use the designated unknown-state key behavior.

Donor classification uses structural graph rules and a hierarchy that supports fallback from a specific donor subclass to broader element or family categories. Haptic groups are detected and represented as one coordination site with multiple participating atoms. The distinction between donor atom and donor group is essential: coordination number counts groups, while the haptic geometry feature uses the member atoms and their centroid or face frame.

## Bond-valence coupling: QV

The optional bond-valence component is defined for sigma metal–donor bonds. For donor distances \(R_i\),

\[
V=\sum_i \exp\!\left(\frac{R_{0,i}-R_i}{B}\right).
\]

The model scores the complex-level sum relative to a reference distribution and subtracts the variance expected if individual Q1 bond lengths were independent. This isolates coupling among the coordination bonds that independent Q1 terms cannot represent. The implementation excludes complexes with haptic sites and is disabled unless its fitted QV/QVR tables are present and selected.

## Coordination constraint objective

The IK optimizer minimizes negative score together with augmented-Lagrangian terms for coordination-shape residuals and symmetry ties. For a nonnegative residual \(v(\mathbf X)\), a representative term is

\[
\mathcal L(\mathbf X,\lambda,\rho)=-S(\mathbf X)+\lambda v(\mathbf X)+\frac{\rho}{2}v(\mathbf X)^2.
\]

The residual is computed after the best-fit proper rotation between assigned ideal site vectors and current donor directions. Multipliers are updated during optimization and can be frozen when residual spread meets the solver's convergence rule. Macrocycle seam closure is applied as a kinematic correction rather than an attractive score term.

## Statistical fitting conventions

Q1 and QO use Gaussian mixtures; Q2 uses Student-t mixtures in cosine space; Q5 uses von Mises mixtures. Mixture complexity is chosen by an information criterion subject to minimum effective support. Sparse keys use partial pooling through an explicit fallback ladder. Structure-level evidence weights prevent a complex with many observations from dominating a key. Details of extraction and fitting are in [training data and refitting](data-and-refit.md).

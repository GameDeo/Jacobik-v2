# Glossary and notation

This file defines the terms used by the technical notes under `paper/`. A term keeps the same meaning in every document. Symbols use SI-derived molecular units, with distances in ångströms and energies in kilojoules per mole unless stated otherwise.

## Chemical and geometric terms

| Term | Definition |
|---|---|
| **Complex** | One metal centre and the ligand instances supplied to a Jacobik solve. |
| **Ligand** | A molecular graph coordinated to the metal through one or more donor groups. |
| **Donor atom** | An atom specified as a metal-binding atom in a ligand input. |
| **Donor group** | The atom or connected set of atoms represented as one coordination unit. A monodentate κ donor has one donor atom; a haptic η donor has multiple participating atoms. |
| **Coordination site** | A vertex in the ideal coordination geometry assigned to one donor group. |
| **Coordination number (CN)** | Number of coordination sites occupied by donor groups. A haptic group occupies one site regardless of the number of participating atoms. |
| **Coordination geometry** | The idealized arrangement of sites around the metal, represented by a named geometry and site vectors. |
| **Assignment** | A bijection from donor groups to coordination sites. |
| **Coordination isomer** | An assignment class after quotienting by proper rotations of the ideal polyhedron and allowed exchanges of identical ligand instances. Reflections are not identified with rotations. |
| **Internal configuration** | A ligand's stereochemistry and discrete internal geometry, such as stereocentre labels, alkene configuration, or a ring pucker. It is distinct from a coordination isomer. |
| **Embed** | A free-ligand Cartesian structure produced before metal coordination is assembled. Bond lengths and most bond angles are subsequently fixed by the kinematic representation. |
| **Rigid body** | A connected fragment of a ligand graph after selected rotatable bonds or bend pivots are represented as joints. |
| **Joint** | A kinematic degree of freedom connecting rigid bodies, typically a torsion or a constrained bend. |
| **Haptic centroid** | The arithmetic centroid of participating atoms used as the point representing a haptic donor group for metal-distance geometry. |
| **Proper rotation** | A rotation in $SO(3)$, with determinant $+1$. It preserves handedness. |
| **Kappa donor** | A donor group coordinated through one atom. |
| **Haptic donor** | A donor group coordinated through a contiguous set of atoms whose combined face or centroid defines one coordination site. |
| **Oxidation state (OS)** | Formal charge assignment to the metal implied by the total charge and ligand charge model; it is an inferred categorical key, not a directly supplied coordinate property. |
| **Spin multiplicity** | Integer label $2S+1$ for the electronic spin state; it is an input and can condition fitted distributions. |
| **Electronic-structure method** | Quantum-chemical approximation used offline to calculate energies, forces, or Hessians. In this project, GFN2-xTB and MACE-POLAR-1 have distinct calibration or evaluation roles. |
| **GFN2-xTB** | Semiempirical tight-binding electronic-structure method used for selected free-ligand refinements, dimer scans, and Hessian calculations. |
| **MACE-POLAR-1** | Machine-learned interatomic potential used for offline distortion calibration and structure-energy evaluation. |
| **ETKDG** | RDKit's distance-geometry conformer generator with torsion preferences. |
| **MMFF** | Merck Molecular Force Field, used for optional free-ligand refinement when parameters are available. |
| **Augmented Lagrangian** | Constraint-optimization method combining multiplier-weighted constraint residuals with quadratic penalties, with multipliers updated during optimization. |
| **B-spline** | Piecewise-polynomial basis function with controlled continuity at knots; tensor products provide smooth multidimensional surfaces. |
| **Density ratio** | Ratio of two probability densities, here reference interaction geometry divided by a matched background density. |
| **Von Mises distribution** | Circular analogue of a Gaussian, parameterized by a mean direction and concentration. |
| **Concentration (kappa)** | Nonnegative parameter controlling the angular localization of a von Mises component. |

## Score and statistics terms

| Term | Definition |
|---|---|
| **Feature** | A scalar or vector descriptor computed from a candidate geometry and scored by a fitted distribution. |
| **Statistical potential** | A score obtained from fitted feature densities. It is an ordinal surrogate for conformer ranking, not a molecular potential-energy surface. |
| **Log-likelihood contribution** | The natural logarithm of a fitted probability density evaluated at a feature. Its unit is the nat. |
| **Nat** | Dimensionless unit of natural logarithmic score. |
| **Mixture component** | One weighted density in a finite mixture, with a component-specific location and scale or covariance. |
| **Mixture weight** | Nonnegative component probability π_k, with ∑_k π_k=1. |
| **Key** | A categorical descriptor used to choose a fitted feature distribution, such as metal, spin, donor subclass, geometry, or topological context. |
| **Fallback ladder** | Ordered sequence of increasingly general keys. Lookup returns the first available fitted entry meeting the fitter's support rule. |
| **Pooling** | Statistical shrinkage of a sparse key's fitted distribution toward a parent key's distribution. |
| **Profile** | Frozen set of result-changing solver and scoring fields, selected by name. |
| **Solve score** | Differentiable score used to optimize candidate coordinates. |
| **Ranking score** | Score used to order completed candidates. Under the `u1` profile these use the same objective definition. |
| **Normalization (Z)** | Per-complex feature-count scale used by the summed objective. It changes score scale but not the argmax for a fixed complex. |
| **Thermal width** | Offline estimate of a feature's standard deviation under harmonic thermal motion around a reference minimum. It is used as a lower bound on fitted component widths. |
| **Evidence weight** | Row weight that makes a training complex contribute a controlled total weight to a fitted key. |
| **Effective sample size** | Sum of evidence weights used as the sample size for mixture selection and support decisions. |

## Graph and contact terms

| Term | Definition |
|---|---|
| **Ligand graph** | Covalent molecular graph with the metal removed. |
| **Bond separation** | Shortest number of ligand-graph edges between two atoms. A missing path means the atoms lie on distinct ligand components. |
| **Contact pair** | An atom pair considered by a nonbonded model after topological exclusions. |
| **Class 4 pair** | A pair separated by four ligand bonds (a 1–5 relationship). |
| **Class 5 pair** | A nonbonded pair beyond the class-4 neighborhood or on different ligands, subject to explicit exclusions. |
| **Metal pair** | A contact pair containing the metal. |
| **Polar hydrogen** | A hydrogen classified as attached to a supported hydrogen-bond donor atom. |
| **Acceptor** | An atom admitted by the relevant NCI candidate rules as an electron-pair acceptor. |
| **NCI opportunity** | A graph-defined pair or ring interaction eligible for a noncovalent interaction model. |
| **Reference structure** | A crystallographic structure used as a training observation or a separately identified reference geometry. |
| **Observation** | A feature value extracted from a reference structure and paired with its categorical key and structure identity. |
| **Candidate row** | One optimization instance for a particular assignment and set of sampled initial conditions. |
| **Beam** | Bounded set of partial torsion seeds retained while expanding combinations of ligand torsion modes. |
| **Basin** | Region of coordinate space associated with one local minimum under a specified energy or score surface. |

## Symbols and units

| Symbol | Meaning |
|---|---|
| $M$ | Metal position. |
| $D_s$ | Donor point for site $s$; for a haptic group this is its centroid. |
| $X_{s,i}$ | Substituent atom $i$ attached to donor group $s$. |
| $e_s$ | Ideal unit vector for coordination site $s$. |
| $\pi$ | Donor-group-to-site assignment map. |
| $r$ | Interatomic or feature distance in Å. |
| θ, φ | Angle or torsion, in radians in equations unless degrees are explicitly shown. |
| $p(x\mid k)$ | Fitted density of feature $x$ under key $k$. |
| $S$ | Aggregate solver score; higher values rank as more favorable. |
| Q1 | Metal–donor distance feature family. |
| Q2 | Donor–metal–donor angle feature family. |
| Q3, Q4 | Legacy donor approach-angle and donor pyramidalization feature families. |
| Q5 | Torsion feature family. |
| Q6 | Nonbonded contact feature family. |
| QO | Joint donor-orientation feature family, also called S_orient. |
| QV | Optional bond-valence coupling feature. |
| $\pi_k$ | Probability weight of mixture component $k$. |
| $\mu_k,\sigma_k$ | Location and standard deviation of a univariate mixture component. |
| $\kappa_k$ | Concentration of von Mises component $k$. |
| $Z$ | Per-complex normalization constant. |
| $R_0,B$ | Bond-valence reference distance and softness parameter. |
| $H$ | Cartesian Hessian at a reference minimum. |
| $H^+$ | Moore–Penrose pseudoinverse of the Hessian on the non-rigid subspace. |
| $k_BT$ | Thermal energy at the stated temperature. |
| $RT$ | Molar thermal energy, $RT=N_Ak_BT$. |
| $SO(3)$ | Group of proper three-dimensional rotations. |
| $C^2$ | Twice continuously differentiable. |
| $C^\infty$ | Infinitely continuously differentiable. |

## Naming conventions

Use **donor group** for the chemical unit and **coordination site** for an ideal geometry vertex. Use **assignment** for their mapping and **coordination isomer** for its symmetry class. Use **internal configuration** for ligand-internal stereochemistry. Use **feature density** or **statistical potential** for the fitted score; reserve **energy** for an energy evaluated by an electronic-structure method. Use **contact pair** for an atom pair and **NCI opportunity** for a candidate eligible for a specific noncovalent model. Use **thermal width** for the Hessian-derived bound and **fit width** for a distribution parameter. Avoid using “isomer,” “potential,” “weight,” or “floor” without the qualifier needed to disambiguate the meaning.

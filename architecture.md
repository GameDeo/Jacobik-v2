# Jacobik v2 Mathematical Architecture

This document translates the Jacobik v2 codebase into a formalized mathematical framework, documenting the end-to-end pipeline, major actions, initializations, mathematical operations, and variable definitions.

## 1. Pipeline Overview

The Jacobik v2 engine generates 3D conformer ensembles for transition metal complexes using an Inverse Kinematics (IK) approach driven by statistical geometry preferences.

The pipeline executes in the following major phases:
1. **Initialization & Shell Construction**: Parse inputs, generate initial 3D embeddings via cheminformatics, and place ligands around a central metal atom.
2. **Topology Decomposition**: Decompose ligands into rigid bodies separated by rotatable bonds to define the kinematic tree.
3. **Statistical Scorer Setup**: Load prior distributions (GMMs/VMMs) for geometric quantities (Q1-Q6) to evaluate conformers.
4. **Seed Generation (Beam Search)**: Explore torsional Degrees of Freedom (DOFs) to find promising initial states.
5. **Joint IK Optimization**: Optimize all DOFs simultaneously using gradient descent (Adam) against a unified objective incorporating a statistical score and a decaying anchor loss.

---

## 2. Initialization & Shell Construction

**File:** `engine.py`

### Inputs
*   **Metal Properties**: Symbol, charge, spin.
*   **Coordination Geometry**: Ideal geometry label (e.g., `octahedral`, `tetrahedral`). Defines $C$ ideal coordination sites (unit vectors $v_c \in \mathbb{R}^3$).
*   **Ligands**: List of $L$ ligands, each defined by a SMILES string and a list of donor atom indices.

### Conformer Generation
For each ligand $l$:
1.  Generate a 3D embedding using RDKit's ETKDGv3 algorithm.
2.  Extract atomic positions $X^{(l)} \in \mathbb{R}^{N_l \times 3}$.

### Placement (`_build_complex_shell`)
1.  Place the metal atom at the origin: $X_{metal} = \vec{0}$.
2.  For each ligand $l$, compute the centroid of its donor atoms $D^{(l)}$:
    $$ \mu_D^{(l)} = \frac{1}{|D^{(l)}|} \sum_{i \in D^{(l)}} X_i^{(l)} $$
3.  Compute an initial target direction for placement:
    $$ \hat{u}^{(l)} = \frac{1}{\|u\|} \begin{bmatrix} \cos(\frac{2\pi l}{L}) \\ \sin(\frac{2\pi l}{L}) \\ 0.3l \end{bmatrix} $$
4.  Translate the ligand such that the donor centroid is placed at a nominal distance of 2.5 Å along the target direction:
    $$ X_i^{(l)} \leftarrow X_i^{(l)} + (2.5 \hat{u}^{(l)} - \mu_D^{(l)}) $$
5.  **Clash Resolution**: Iteratively push the ligand outward along the direction from the metal to the closest donor until the minimum metal-donor distance is $\ge 1.2$ Å.

---

## 3. Topology Decomposition

**File:** `topology/decompose.py`, `ik/fk.py`

### Kinematic Tree Definition
To enable Inverse Kinematics, each ligand is decomposed into a kinematic tree:
1.  **Graph Construction**: Build an undirected graph $G$ from ligand internal bonds (bond order $\ge 0.5$).
2.  **Rotatable Bonds**: Identify rotatable bonds as edges in $G$ with bond order $1.0$ that are not part of any ring (found via `minimum_cycle_basis`).
3.  **Rigid Bodies**: Removing all rotatable bonds partitions $G$ into connected components. Each component is a **Rigid Body**.
4.  **Pivot Chains**: A BFS from the body containing donor atoms establishes a tree of rigid bodies connected by **Pivot Joints** (rotatable bonds).

### Degrees of Freedom (DOFs)
For a complex, the DOFs are parameterized by $\theta \in \mathbb{R}^{B \times \sum_l (6 + J_l)}$, where $B$ is the batch size and $J_l$ is the number of pivot joints in ligand $l$.
For a single ligand, $\theta^{(l)} = [t, \omega, \phi]$:
*   **Base Translation** $t \in \mathbb{R}^3$.
*   **Base Rotation** $\omega \in \mathbb{R}^3$ (axis-angle representation).
*   **Torsions** $\phi \in \mathbb{R}^{J_l}$ (absolute joint angles in radians).

### Forward Kinematics (FK)
**File:** `ik/fk.py`, `ik/kernels.py`

The FK map $\mathcal{F}(\theta, X_{ref})$ computes updated atom positions via batched tensor operations over batch size $B$ and atoms $N$:
1.  **Base Transform**:
    Converts the base rotation $\omega \in \mathbb{R}^{B \times 3}$ to rotation matrices $R \in \mathbb{R}^{B \times 3 \times 3}$ using a batched implementation of Rodrigues' rotation formula (`axisangle_to_matrix`):
    *   For $\|\omega\| \ge \epsilon$: $R = I\cos\theta + [\hat{k}]_\times \sin\theta + \hat{k}\hat{k}^T(1-\cos\theta)$, where $\theta = \|\omega\|$ and $\hat{k} = \omega/\theta$.
    *   For $\|\omega\| < \epsilon$ (Taylor expansion fallback to ensure finite gradients): $R \approx I + [\omega]_\times$, where $[\omega]_\times$ is the skew-symmetric matrix of $\omega$.
    Let $\mu_{piv} \in \mathbb{R}^{B \times 3}$ be the centroid of the ligand's donor atoms. 
    The positions of the ligand atoms $X_{lig} \in \mathbb{R}^{B \times N_l \times 3}$ are transformed via batched matrix multiplication (`torch.bmm`):
    $$ X_{base} = \text{bmm}(X_{lig} - \mu_{piv}, R^T) + \mu_{piv} + t $$
2.  **Torsional Updates** (`rodrigues` kernel):
    For each joint $k \in \{1 \dots J_l\}$, defined by atoms $p_j$ to $p_k$:
    *   Compute current dihedral angle $\phi_{cur}$ via cross products and dot products of the vectors flanking the rotatable bond.
    *   Calculate required rotation $\Delta\phi_k = \phi_k - \phi_{cur}$.
    *   Let $\hat{k} = \frac{p_k - p_j}{\|p_k - p_j\|}$ be the rotation axis. For the subset of downstream atoms $V \in \mathbb{R}^{B \times N_{down} \times 3}$, we translate them relative to pivot $p_j$, then apply the batched Rodrigues kernel (using broadcasting and `torch.linalg.cross`):
        $$ V_{rot} = V\cos\Delta\phi_k + (\hat{k} \times V)\sin\Delta\phi_k + \hat{k}(\hat{k} \cdot V)(1-\cos\Delta\phi_k) $$
    *   The updated subset is then translated back: $X_{downstream} \leftarrow V_{rot} + p_j$.

---

## 4. Statistical Scorer Objective

**File:** `ik/scorer.py`

The objective evaluates conformers based on statistical preferences mined from the CSD. The total statistical score is an aggregate of six quantities (Q1-Q6).

### Quantities & Distributions
*   **Q1: Metal-Donor Distances (GMM)**
*   **Q2: Donor-Metal-Donor Angles (VMM - von Mises Mixture)**
*   **Q3: Approach Angles (VMM)**
*   **Q4: Donor Pyramidalization (GMM)**
*   **Q5: Ligand Torsions (VMM)**
*   **Q6: Steric Clashes (Half-Gaussian Penalty)**

### Batched Distribution Probabilities
Evaluating geometric quantities across batches ($B$), atoms/pairs ($N$), and mixture components ($K$) is fully vectorized. For an observed tensor $x \in \mathbb{R}^{B \times N}$:

**Gaussian Mixture Model (GMM)** (Q1, Q4):
To evaluate $P_{GMM}(x)$, the engine precomputes effective variances $\sigma_{eff}^2 = \max(\sigma^2, \sigma_{min}^2)$ and weights $W = \log(w) + \log(q_{weight})$. The log-probabilities over $K$ components are aggregated via a numerically stable log-sum-exp over the last dimension:
$$ \log P_{GMM}(x) = \text{logsumexp}_K \left( -0.5\log(2\pi\sigma_{eff}^2) - \frac{(x - \mu)^2}{2\sigma_{eff}^2} + W \right) $$

**Von Mises Mixture (VMM)** (Q2, Q3, Q5):
To evaluate $P_{VMM}(x)$, the normalizer $\log(2\pi I_0(\kappa))$ is precomputed using `torch.special.i0e` (exponentially scaled modified Bessel function) to prevent overflow for large $\kappa$:
$$ \log N_{norm} = \log(2\pi) + \log(I_0^e(\kappa)) + \kappa $$
$$ \log P_{VMM}(x) = \text{logsumexp}_K \left( \kappa \cos(x - \mu) - \log N_{norm} + W \right) $$

**Half-Gaussian (Q6)**:
Evaluates distance $d$ for a pair of atoms with van der Waals boundary $\mu$ and tolerance $\sigma$.
$$ \text{Penalty}(d) = \begin{cases} -\frac{(d - \mu)^2}{2\sigma^2} & \text{if } d < \mu \\ 0 & \text{otherwise} \end{cases} $$

### Total Score $S(X)$
The raw geometric observations are extracted via `GeometryExtractor` evaluated on the FK-updated positions $X$. The total score is a weighted sum of the mean normalized log-probabilities for Q1-Q5, and a soft-min aggregated penalty for Q6.

$$ S(X) = \sum_{q \in \{1..5\}} W_q \left( \frac{1}{N_q} \sum_{i=1}^{N_q} \log P_q(x_i) \right) + W_6 \left( -\frac{1}{\tau} \text{LSE}\left(-\tau \cdot \text{Penalty}_{Q6}(d_{pairs})\right) + \frac{\log(N_{pairs})}{\tau} \right) $$

*(The Q6 soft-min with temperature $\tau=2.0$ bridges the mean penalty for uniform violations and the max penalty for severe single clashes).*

---

## 5. Beam Search (Seed Generation)

**File:** `ik/beam.py`

To find good starting states for the optimization, `BeamSearch` explores the Q5 (torsional) space.
For a ligand with $J$ joints, maintaining a beam of width $W$:
1.  **Iteration**: For joint $k=1 \dots J$:
    *   Expand each of the $W$ current candidates into $K$ new candidates, where $K$ is the number of mixture components in the Q5 VMM for joint $k$.
    *   Set the new candidate's $\phi_k$ to the component mean $\mu_k$.
    *   Evaluate a partial score using Q5 and a partial Q6 (checking clashes between the newly moved rigid body and all previously fixed bodies).
2.  **Pruning**: Sort the $W \times K$ expanded candidates by score and retain the top $W$ unique prefixes.
3.  **Result**: Yields $W$ seed torsion vectors $\phi_{seed} \in \mathbb{R}^J$.

---

## 6. Joint IK Optimization

**File:** `ik/ik_solver.py`

### Kabsch Initialization
For a given assignment of donor atoms to ideal geometry sites $Y_{ideal} \in \mathbb{R}^{|D| \times 3}$:
The base translational and rotational DOFs $[t, \omega]$ are initialized using the Kabsch algorithm to align the current donor positions $P$ to the target sites $Q=Y_{ideal}$. This minimizes the Frobenius norm $\|RP + t - Q\|_F^2$.
1.  Center both point sets: $P_c = P - \mu_P$, $Q_c = Q - \mu_Q$.
2.  Compute cross-covariance matrix: $H = P_c^T Q_c \in \mathbb{R}^{3 \times 3}$.
3.  Perform Singular Value Decomposition (SVD): $U, S, V^T = \text{SVD}(H)$.
4.  Compute rotation $R = V U^T$. To prevent reflection, we check if $\det(V U^T) < 0$, and if so, flip the sign of the last column of $V$.
5.  Extract translation $t = \mu_Q - R \mu_P$ and convert $R$ to axis-angle $\omega$.

If the donor set is degenerate (e.g., single point or collinear), the cross-covariance $H$ is rank-deficient. Orientational diversity is injected across the batch by computing eigenvectors of the unconstrained axes and applying random rotations.

### Unified Objective Function
The optimizer minimizes a combined loss function blending the statistical score and an anchor loss over $T$ steps:
$$ \mathcal{L}(\theta) = -S(\mathcal{F}(\theta, X_{ref})) + \lambda(t) \cdot \mathcal{L}_{anchor}(\theta) $$

**Anchor Loss ($\mathcal{L}_{anchor}$)**:
Ensures the donors stay near the ideal geometry sites during the initial phases. Employs a Log-Sum-Exp (LSE) soft-max over squared errors to focus gradients on the worst-placed donor.
$$ E_i(\theta) = \| X_{donor\_i}(\theta) - Q_i \|^2 $$
$$ \mathcal{L}_{anchor}(\theta) = \frac{1}{\tau_{lse}} \log \left( \sum_i \exp(\tau_{lse} \cdot E_i(\theta)) \right) $$

**Anchor Schedule ($\lambda(t)$)**:
Init-only semantic decay. The anchor is initially strong ($\lambda_{init}$) and decays quadratically toward zero as the statistical score $S(\theta)$ exceeds its initial value $S_0$.
$$ \text{gain}(t) = \max(0, S(\theta_t) - S_0) $$
$$ \lambda(t) = \lambda_{init} \cdot \left( \max\left(0, 1 - \frac{\text{gain}(t)}{\lambda_{init}}\right) \right)^2 $$

### Optimization Algorithm
The optimization uses a single batched **Adam** optimizer on $\theta$.
*   **Gradient Computation**: $\nabla_\theta \mathcal{L}(\theta)$ is computed via a single PyTorch `autograd.grad` pass. By combining the anchor and statistical losses into a single scalar expression before `backward()`, the engine avoids multiple graph traversals.
*   **Null-Space Projector for Rotations**: For an underdetermined ligand (e.g., bidentate with 2 donors), rotation around the axis joining the donors leaves the donors stationary, rendering the anchor's gradient along that axis redundant or noisy. 
    The engine computes a batched projection matrix $P_{keep} \in \mathbb{R}^{B \times 3 \times 3}$:
    1. Let $v_i = X_{donor\_i} - \mu_{donor}$ be the centered donor positions.
    2. Construct the skew-symmetric matrix $J = [v_1]_\times \dots [v_D]_\times$ and perform SVD: $U, S, V^T = \text{SVD}(J)$.
    3. Construct $P_{keep} = \sum_{k: S_k \ge tol} V_k V_k^T$. (For exactly two donors, this yields $P_{keep} = I_{3 \times 3} - \hat{n}\hat{n}^T$ where $\hat{n}$ is the donor-donor axis).
    The gradient of the base rotation is then projected: $\nabla_\omega \mathcal{L} \leftarrow \text{bmm}(P_{keep}, \nabla_\omega \mathcal{L})$.
*   **Axis-Angle Renormalization**: Post-step, if $\|\omega\| > \pi$, it is renormalized to the equivalent minimal rotation $\omega \leftarrow \omega \cdot \frac{\pi}{\|\omega\|}$ to bound the parameter space.

### Convergence
Convergence is determined via a rolling plateau check.
If the range of $S(\theta)$ over the last $W_{plateau}$ steps is less than $\epsilon_{atol}$ (e.g., 0.02 nats), the trajectory is considered converged.
The solver tracks and returns the DOFs that achieved the highest honest statistical score $S(\theta)$ across the entire trajectory.

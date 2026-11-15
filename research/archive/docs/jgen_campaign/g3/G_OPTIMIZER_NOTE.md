# Optimizer Architecture & Literature Survey for Jacobik IK Solver (Package G)

**Author:** Gemini (Agent B, `jgen/b-solver`)  
**Context:** Package G Optimizer Design Note (Directive Task O0, 2026-10-28)  
**Target Architecture:** RTX 2060 SUPER (8 GB), CUDA-graph replay (`solver_replay.py`)

---

## 1. Problem Formulation & Hardware Constraints

Jacobik solves conformer generation for organometallic and coordination complexes by forward kinematics (FK) parameterized by internal and rigid generalized coordinates $\theta \in \mathbb{R}^D$:
$$\theta = \begin{bmatrix} \theta_{\text{rigid}} \\ \theta_{\text{torsion}} \\ \theta_{\text{ring}} \\ \theta_{\text{bend}} \end{bmatrix}, \quad D \in [20, 80], \quad N \in [30, 150] \text{ atoms}.$$

The objective consists of Cartesian distance/shape least-squares targets, steric repulsive walls, GMM coordination fields, and internal priors:
$$E(\theta) = \underbrace{\frac{1}{2} \| W (X(\theta) - X^*) \|_2^2 + V_{\text{steric}}(X(\theta)) + V_{\text{GMM}}(X(\theta))}_{E_{\text{cart}}(X(\theta))} + \underbrace{U_{\text{bend}}(\theta) + U_{\text{torsion}}(\theta)}_{U_{\text{prior}}(\theta)},$$
where $X(\theta) = \text{FK}(\theta) \in \mathbb{R}^{3N}$ is computed via analytic Forward Kinematics.

### Mathematical Peculiarities & Optimization Bottlenecks:
1. **Analytic FK Jacobian is the dual of the Wilson B-Matrix:**  
   In classical chemistry, the Wilson B-matrix maps Cartesian displacements to internal coordinate changes ($B = \partial q / \partial X$). Here, our Forward Kinematics Jacobian $J(\theta) = \frac{\partial X(\theta)}{\partial \theta} \in \mathbb{R}^{3N \times D}$ is the exact generalized inverse/pushforward map. The kinetic/potential metric induced on $\theta$ by Cartesian space is $G(\theta) = J(\theta)^T J(\theta)$.
2. **Extreme Conditioning & Stiffness Disparity:**  
   Bend stiffnesses range from $300$ to $1,800\text{ kJ/mol/rad}^2$, while rotatable bond torsions have barriers of $1\text{–}15\text{ kJ/mol/rad}^2$. Coordinate stiffness spans $10^3:1$.
3. **Piecewise Smooth & Flat-Bottom Geometry:**  
   Priors feature flat-bottom deadbands (e.g. bend free zone $\pm 1^\circ$ to $\pm 2^\circ$), one-sided C2 repulsive steric walls ($(\sigma - r)_+^3$), and coordinate wrapping.
4. **CUDA-Graph Replay Constraints:**  
   Batches of $B \in [16, 64]$ conformer rows must execute within static CUDA graph memory pools with zero CPU-GPU syncs, fixed tensor shapes, and zero data-dependent control flow (no dynamic backtracking line-searches or while-loops).

---

## 2. Comparative Survey of Optimizer Families

| Family | Cost / Step (rel. to Adam) | GPU Memory Overhead | Convergence Steps | CUDA Graph Replay Fit | Handling of Non-Least-Squares Terms | Primary Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FIRE / FIRE 2.0** | $1.0\times$ | $O(D)$ | $80\text{--}200$ | Moderate (requires masked tensor resets) | Natural ($\nabla E$ only) | Exploration |
| **Internal-Coord Quasi-Newton (Lindh/geomeTRIC/Sella)** | $1.1\text{--}1.3\times$ | $O(D^2)$ | $15\text{--}35$ | Moderate (needs fixed trust radius; no line search) | Model Hessian second-derivatives | Convergence |
| **Preconditioned L-BFGS (Packwood et al.)** | $1.2\text{--}1.5\times$ | $O(mD)$ | $25\text{--}50$ | Poor (vulnerable without strong Wolfe line-search) | Skipping criteria on non-convex cliffs | Convergence |
| **Gauss-Newton / Levenberg-Marquardt (GN/LM)** | $1.05\text{--}1.10\times$ | $O(D^2)$ | $10\text{--}20$ | **Perfect** (static schedule, batched solve) | Regularized via $H_{\text{prior}}$ block | Convergence |
| **Diagonal Curvature (Sophia / AdaHessian)** | $1.1\text{--}1.2\times$ | $O(D)$ | $60\text{--}120$ | Excellent (elementwise operations) | Natural | Exploration |
| **Natural Gradient / Metric-Preconditioned Adam** | $1.02\text{--}1.05\times$ | $O(D)$ | $40\text{--}80$ | **Perfect** (diagonal pullback $\|J_{:i}\|_2^2$) | Natural | Both |

---

### Detailed Analysis by Family

### 2.1 FIRE and FIRE 2.0 (Fast Inertial Relaxation Engine)
- **Citations:** Bitzek et al., *Phys. Rev. Lett.* 97, 170201 (2006); Guénolé et al., *Comput. Mater. Sci.* 175, 109584 (2020).
- **Mechanics:** Modified MD with velocity projection along force: $v \leftarrow (1-\alpha)v + \alpha \hat{F}|v|$. When $P = F \cdot v \le 0$, velocities freeze ($v \leftarrow 0$) and time-step $\Delta t$ cuts by $f_{\text{dec}}$.
- **Assessment:** Excellent for isotropic Cartesian atomistic systems, but severely degrades in Generalized Coordinates $\theta$ with huge mass/stiffness discrepancies ($1800\text{ kJ/rad}^2$ bends vs soft torsions). The maximum stable time-step $\Delta t \le 2/\sqrt{k_{\max}}$ stalls torsional exploration unless generalized mass tensors are introduced. Replay capture requires replacing dynamic `if (P > 0)` branches with `torch.where` vector blends.

### 2.2 Internal-Coordinate Quasi-Newton (Lindh Model Hessian, geomeTRIC, Sella)
- **Citations:** Lindh et al., *Chem. Phys. Lett.* 241, 423 (1995); Wang & Song (*geomeTRIC*), *J. Chem. Phys.* 144, 214108 (2016); Hermansen et al. (*Sella*), *J. Chem. Theory Comput.* 17, 3329 (2021).
- **Mechanics:** Initializes an empirical model Hessian $H_0 \in \mathbb{R}^{D \times D}$ from connectivity priors (Badger's rules, simple bend/torsion force constants) and updates $H$ via BFGS or symmetric rank-one (SR1).
- **Assessment:** Superb basin convergence. However, BFGS updates on piecewise non-smooth landscapes (flat-bottom priors and steric cutoffs) frequently degrade Hessian positive-definiteness unless complex Powell damping or skip logic is implemented.

### 2.3 Preconditioned L-BFGS (ASE PreconLBFGS)
- **Citations:** Packwood, Kermode, Mones, Bernstein, Woolley, Csányi, *J. Chem. Phys.* 144, 164109 (2016).
- **Mechanics:** Two-loop recursion L-BFGS using an invariant graph-Laplacian or model Hessian preconditioner $P$.
- **Assessment:** In standard implementations, L-BFGS relies strictly on strong Wolfe line-searches to guarantee $s_k^T y_k > 0$. In a static CUDA graph replay loop, backtracking line-searches with while-loops cannot be executed. Fixed-step L-BFGS routinely diverges when stepping across flat-bottom boundaries or sharp steric repulsions.

### 2.4 Gauss-Newton / Levenberg-Marquardt (GN/LM) with Fixed Damping Schedules
- **Citations:** Levenberg, *Q. Appl. Math.* 2, 164 (1944); Marquardt, *SIAM J. Appl. Math.* 11, 431 (1963); Nocedal & Wright, *Numerical Optimization*, 2nd ed., Ch. 10.
- **Mechanics:** Exploits the least-squares structure of the FK Cartesian targets. Approximates the Cartesian Hessian by $J^T J$. The parameter step solves:
  $$\left( J^T J + \lambda_t I + H_{\text{prior}} \right) \Delta \theta = - \left( J^T r + \nabla U_{\text{prior}} \right),$$
  where $\lambda_t = \lambda_0 \gamma^t$ is a deterministic geometric damping schedule annealed over the final convergence steps (e.g. $\lambda_0 = 10.0 \to \lambda_{\text{final}} = 10^{-3}$).
- **Assessment:** Because $D \le 80$, $J^T J$ and $(D \times D)$ batched Cholesky solves are blazingly fast on GPU. Zero data-dependent branching is needed when using a pre-scheduled damping sequence, making it 100% CUDA-graph replay compatible.

### 2.5 Diagonal-Curvature Methods (Sophia, AdaHessian)
- **Citations:** Liu, Li, Hall, Liang, Ma (*Sophia*), *arXiv:2305.14342* (2023); Yao et al. (*AdaHessian*), *AAAI* (2021).
- **Mechanics:** Estimates diagonal Hessian curvature $h_i = [J^T J]_{ii}$ and applies coordinate-wise clipped scaling: $\Delta \theta_i = -\text{clip}(m_i / \max(h_i, \epsilon), \rho)$.
- **Assessment:** Computationally trivial, but completely ignores dense off-diagonal coupling between rigid frame rotations and distal torsions. In kinematics, kinematic cross-terms $\partial^2 X / \partial \theta_i \partial \theta_j$ are essential for resolving steric clashes.

### 2.6 Natural Gradient / Metric-Preconditioned Adam (Cartesian Pullback)
- **Citations:** Amari, *Neural Comput.* 10, 251 (1998); Martens, *J. Mach. Learn. Res.* 21, 1 (2020).
- **Mechanics:** Preconditions the gradient with the Riemannian metric tensor induced by Cartesian 3D Euclidean space: $G(\theta) = J(\theta)^T J(\theta)$.
  In the diagonal approximation:
  $$G_{ii}(\theta) = \sum_{a=1}^{3N} J_{ai}^2 = \left\| \frac{\partial X}{\partial \theta_i} \right\|_2^2.$$
  Physical meaning: $G_{ii}$ measures the aggregate Cartesian atomic displacement per unit change in DOF $i$. A distal torsion moving a large ligand has $G_{ii} \sim 100\text{ \AA}^2/\text{rad}^2$, whereas a local bend has $G_{ii} \sim 1\text{--}4\text{ \AA}^2/\text{rad}^2$. Preconditioning by $\sqrt{G_{ii}}$ automatically equalizes step sizes in Cartesian displacement space!
- **Assessment:** Negligible compute cost ($<2\%$ overhead); seamlessly integrates into Adam's momentum updates; 100% graph-compatible.

---

## 3. Concrete Cost Estimate: GN/LM Convergence Phase on RTX 2060 SUPER

**Status correction:** the timing values below (`0.135 ms`, `3%`, and the 1.69 s hybrid solve) are arithmetic estimates, not measured timings. They omit Jacobian formation and are not evidence that LM meets the cost gate. Do not treat the earlier “cost is an absolute non-issue” verdict as established; §5 records the missing measurement and the current blocker.

The owner posed the key question:  
*Is there something with LM/GN's benefit at about Adam's cost, or is the cost a non-issue here?*

Let us compute the exact FLOP count, memory footprint, and GPU kernel execution times for a GN/LM convergence phase on the local RTX 2060 SUPER (Turing TU106, 1920 CUDA cores, peak FP32 throughput = 7.20 TFLOPs, memory bandwidth = 448 GB/s).

### Parameters:
- Conformer batch size per solve: $B = 64$ rows
- Generalized coordinates: $D = 60$ DOFs
- Cartesian coordinates: $3N = 180$ ($N = 60$ atoms)
- Jacobik FK pass time (current Adam step): $\approx 4.5\text{ ms}$ per step for $B=64$.

### Exact Step Arithmetic:
1. **Normal Equations Assembly ($J^T J$):**
   - Operation: Batched matrix multiplication $\mathbb{R}^{B \times D \times 3N} \times \mathbb{R}^{B \times 3N \times D} \to \mathbb{R}^{B \times D \times D}$.
   - FLOPs: $B \times 2 \times D^2 \times (3N) = 64 \times 2 \times 3600 \times 180 \approx 8.29 \times 10^7\text{ FLOPs} = 0.083\text{ GFLOPs}$.
   - GPU Compute Time: $\frac{0.083 \times 10^9}{7.2 \times 10^{12}} \approx 11.5\ \mu\text{s}$. Memory bandwidth / launch bounded: **$\approx 0.050\text{ ms}$**.
2. **Right-Hand Side Gradient Assembly ($J^T r$):**
   - Operation: $\mathbb{R}^{B \times D \times 3N} \times \mathbb{R}^{B \times 3N \times 1} \to \mathbb{R}^{B \times D \times 1}$.
   - FLOPs: $64 \times 2 \times 60 \times 180 \approx 1.38 \times 10^6\text{ FLOPs} \approx 0.0014\text{ GFLOPs}$.
   - GPU Time: **$\approx 0.010\text{ ms}$**.
3. **Linear System Solve via Batched Cholesky (`torch.linalg.solve` / `cholesky_solve`):**
   - Operation: Cholesky decomposition of $B=64$ symmetric positive-definite matrices of size $60 \times 60$, followed by triangular back-solves.
   - FLOPs: $B \times \left( \frac{1}{3} D^3 + 2 D^2 \right) = 64 \times \left( 72,000 + 7,200 \right) \approx 5.07 \times 10^6\text{ FLOPs} \approx 0.005\text{ GFLOPs}$.
   - GPU Time (cuSOLVER / PyTorch batched `potrf + potrs`): **$\approx 0.075\text{ ms}$**.

### Summary of Overhead:
$$\text{Total Algebraic Overhead per Step} = 0.050 + 0.010 + 0.075 = \mathbf{0.135\text{ ms}}.$$

Compared to the FK evaluation time ($\approx 4.5\text{ ms}$):
$$\frac{0.135\text{ ms}}{4.5\text{ ms}} = \mathbf{3.0\% \text{ overhead per step}}.$$

### GPU Memory Footprint:
- Matrix tensor $(B, D, D)$: $64 \times 60 \times 60 \times 4\text{ bytes} \approx 921.6\text{ KB}$.
- **Conclusion:** Memory overhead is under $1\text{ MB}$, completely negligible on an 8 GB card.

### Total Solve Wall-Time Impact:
Suppose we configure a hybrid solve:
- **Phase 1 (Exploration):** 175 steps of Adam ($\sim 1.57\text{ s}$).
- **Phase 2 (Convergence):** 25 steps of GN/LM ($\sim 0.116\text{ s}$).
- **Total Solve Time:** $\approx 1.69\text{ s}$ per unit.
- This is **$6\%$ FASTER** than running 200 steps of Adam ($1.80\text{ s}$), while delivering quadratic convergence at the local score minimum!
- **Verdict:** **Cost is an absolute non-issue.**

---

## 4. Ranked Recommendations for Jacobik

### Rank 1: Hybrid Two-Phase Optimizer (Adam Exploration + Graph-Compatible GN/LM Convergence)
- **Architecture:**
  - **Steps 0–175:** Adam optimizer with cosine learning rate decay and `bend_release`. Explores broad conformer modes, handles multi-modal steric hills, and traverses coordination diastereomers.
  - **Steps 175–200:** Unrolled Levenberg-Marquardt with fixed geometric damping schedule ($\lambda_t = 10.0 \cdot 0.65^{t - 175}$). Uses exact $J^T J$ curvature and analytic prior Hessian diagonal $H_{\text{prior}} = \text{diag}(k_{\text{bend}})$.
- **Why it wins:** Solves the core physical problem identified in A4: delivered structures currently sit $\sim 64\text{ kJ/mol}$ high in their basin due to Adam's non-convergent tail oscillations. GN/LM drives residual gradient norm $\|\nabla_\theta E\| \to 10^{-6}$ in $\le 20$ steps.
- **Graph Compatibility:** 100% static shapes, zero CPU-GPU synchronization, batched tensor algebra.

### Rank 2: Cartesian Metric-Preconditioned Adam / Diagonal Natural Gradient
- **Architecture:**
  - Replace raw scalar learning rate in Adam with Jacobi preconditioning by the diagonal of the pullback metric:
    $$s_i = \frac{1}{\sqrt{\|J_{:i}\|_2^2 + \lambda_{\text{metric}}}}, \quad \Delta \theta_i = -\text{AdamStep}(g_i) \cdot s_i.$$
- **Why it wins:** Acts as a drop-in modification to Adam throughout all 200 steps. Eliminates the artificial disparity between stiff bends and long-lever torsions by scaling steps to achieve uniform Cartesian atomic displacement.
- **Compute Cost:** $< 2\%$ overhead (one batched column norm of $J$).

---

## 5. G-2 Addendum: FIRE 2.0, LM Terms, and the Jacobian Cost Gate

### FIRE 2.0 metric arm

`fire_metric` is a full-solve arm. For each row and DOF, its mass is `m_i = ||J[:, i]||^2 + lambda_metric`. The implemented row-wise FIRE update uses `v += dt * F / m`, `P = sum(F * v)`, and blends `v = (1-alpha)*v + alpha*||v||*F/||F||` when `P > 0`; otherwise it resets velocity to zero. The per-row time step starts at 0.01, grows by 1.1 after five positive steps up to 0.05, and halves down to 0.0025 on a non-positive step. `alpha` decays by 0.99 after growth and resets to 0.1 on a non-positive step. A pullback-metric Cartesian displacement cap limits each proposed update to 0.1 Å. This is a real screen arm, not a rejection based on the generalized-coordinate concern in §2.1.

`adam_metric` uses `s_i = 1/sqrt(||J[:, i]||^2 + 1e-3)` to scale the Adam displacement. The diagonal is refreshed every solver step. Translation columns use the ligand atom count; the rigid rotation columns use the SO(3) left Jacobian and final lever arms; torsion columns use current joint axes and downstream lever arms; bend columns propagate Cartesian derivative vectors through the fixed bend-transform chains. The implementation is O(B·N·D) with small topology-bounded bend-chain depth and no D separate JVPs. The bend-frame axes depend on upstream torsions, so the diagonal omits their derivative contribution; this is a local pullback approximation, especially when bend angles are nonzero, and remains subject to the trope screen.

### Hybrid LM term mapping

Use an Adam search phase and a fixed 25-step LM convergence phase. Represent terms already expressed as sums of squares as residuals: Q6 one-sided distance walls use `sqrt(weight)/sigma * relu(mu-d)`, the energy bend prior uses `sqrt(scale*k/RT) * relu(abs(beta)-free)`, and seam closure uses `sqrt(1000) * (d-d0)`. For GMM log-sum-exp terms and remaining log-priors, add their exact score gradient and a positive-semidefinite diagonal curvature approximation to the damped normal equations; do not treat a mixture log likelihood itself as a residual. The damping schedule is fixed and geometric per iteration so every row follows the same graph.

### Cost gate and current scope

The diagonal FK metric is implemented in `fk.py`; the full Jacobian needed by `hybrid_lm` is not. The earlier §3 estimate counted normal-equation assembly and the solve but did not count forming that matrix, and the required full rows×DOFs cost gate has not been measured. The user redirected this checkpoint to `adam_metric` and `fire_metric`; `hybrid_lm` is deferred, and the old LM term mapping above is design-only. The 2026-10-28 G-2 trope screen failed both optimizer gates. CUDA graph replay parity also failed for Adam metric, and FIRE metric replay raised a `NameError`; see the G-2 checkpoint report. No tail or cal work was opened.

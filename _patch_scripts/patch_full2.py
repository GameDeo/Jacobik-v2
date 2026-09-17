#!/usr/bin/env python3
import sys

PATH = "src/jacobik/ik/ik_solver.py"

with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

steps_done = []


def do_replace(label, old, new, count=1):
    global src
    n = src.count(old)
    if n == 0:
        print(f"FATAL: pattern for step [{label}] not found", file=sys.stderr)
        sys.exit(1)
    if n != count:
        print(f"FATAL: pattern for step [{label}] found {n} times, expected {count}", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, count)
    steps_done.append(label)


# ---------------------------------------------------------------------------
# Step 3: class docstring param additions
# ---------------------------------------------------------------------------
OLD_DOC = '''    anchor_floor_frac : fraction of lambda_init retained permanently instead
                   of decaying fully to zero (default 0.0 = old behaviour).
                   EDTA-class crowded/flexible polydentate ligands were
                   observed to drift to a WORSE score and geometry under a
                   longer solve once the anchor fully vanished (Q1-Q6 alone
                   underconstrains the donor shell once nothing is pinning
                   it to the target polyhedron) -- a small nonzero floor
                   (~0.02-0.05) keeps a gentle permanent pull toward the
                   assigned geometry without preventing the statistical
                   score from doing the local refinement.
    device       : torch device
    """'''
NEW_DOC = '''    anchor_floor_frac : fraction of lambda_init retained permanently instead
                   of decaying fully to zero (default 0.0 = old behaviour).
                   EDTA-class crowded/flexible polydentate ligands were
                   observed to drift to a WORSE score and geometry under a
                   longer solve once the anchor fully vanished (Q1-Q6 alone
                   underconstrains the donor shell once nothing is pinning
                   it to the target polyhedron) -- a small nonzero floor
                   (~0.02-0.05) keeps a gentle permanent pull toward the
                   assigned geometry without preventing the statistical
                   score from doing the local refinement.
                   DEPRECATED under the 2.3.0 shape-ALM path (solve() called
                   with ideal_directions given): the dual multiplier
                   mu_shape already re-inflates whenever the constraint is
                   violated again, so a manual floor is unnecessary there.
                   Still honoured on the old fixed-schedule fallback path.
    shape_rho    : Augmented-Lagrangian penalty/step-size rho for the shape
                   constraint dual ascent (2.3.0+; only used when solve() is
                   called with ideal_directions). mu_shape <- clamp(mu_shape
                   + shape_rho * violation, min=0) every step. (default 8.0)
    shape_tol    : shape-deviation tolerance below which the constraint is
                   considered satisfied and contributes no violation
                   (2.3.0+; units: mean/LSE per-donor squared unit-vector
                   residual after optimal rotation -- a value of ~0.02
                   corresponds to roughly an 8-degree worst-donor deviation
                   from the ideal polyhedron. Chosen to match how CShM is
                   used in the inorganic chemistry literature: near-zero
                   deviation isn't required, only "close enough to be the
                   same shape family". Starting default, not empirically
                   fit -- tune against the test-suite notebook.)
                   (default 0.02)
    sym_weight   : fixed weight on the symmetry-tying restraint that pulls
                   topologically-equivalent donors' M-D distances toward
                   their group mean (2.3.0+; see engine.py's RDKit
                   canonical-rank symmetric-donor detection). This is a soft
                   chemical prior (analogous to crystallography's SIMU/SAME
                   restraints), not a hard constraint, so it uses a fixed
                   weight rather than the dual-ascent machinery above.
                   (default 2.0)
    device       : torch device
    """'''
do_replace("3-class-docstring", OLD_DOC, NEW_DOC)

# ---------------------------------------------------------------------------
# Step 4: __init__ signature + body
# ---------------------------------------------------------------------------
OLD_INIT = '''    def __init__(
        self,
        fks: list,
        scorer: "Scorer",
        ge: "GeometryExtractor",
        n_steps: int = 200,
        lr: float = 0.01,
        lambda_init: float = 10.0,
        lse_temp: float = 2.0,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        plateau_window: int = 8,
        plateau_atol: float = 0.02,
        anchor_floor_frac: float = 0.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.fks = fks
        self.scorer = scorer
        self.ge = ge
        self.n_steps = n_steps
        self.lr = lr
        self.lambda_init = lambda_init
        self.lse_temp = lse_temp
        self.adam_beta1, self.adam_beta2 = adam_betas
        self.adam_eps = 1e-8
        self.plateau_window = plateau_window
        self.plateau_atol = plateau_atol
        self.anchor_floor_frac = anchor_floor_frac
        self.device = device'''
NEW_INIT = '''    def __init__(
        self,
        fks: list,
        scorer: "Scorer",
        ge: "GeometryExtractor",
        n_steps: int = 200,
        lr: float = 0.01,
        lambda_init: float = 10.0,
        lse_temp: float = 2.0,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        plateau_window: int = 8,
        plateau_atol: float = 0.02,
        anchor_floor_frac: float = 0.0,
        shape_rho: float = 8.0,
        shape_tol: float = 0.02,
        sym_weight: float = 2.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.fks = fks
        self.scorer = scorer
        self.ge = ge
        self.n_steps = n_steps
        self.lr = lr
        self.lambda_init = lambda_init
        self.lse_temp = lse_temp
        self.adam_beta1, self.adam_beta2 = adam_betas
        self.adam_eps = 1e-8
        self.plateau_window = plateau_window
        self.plateau_atol = plateau_atol
        self.anchor_floor_frac = anchor_floor_frac
        self.shape_rho = shape_rho
        self.shape_tol = shape_tol
        self.sym_weight = sym_weight
        self.device = device'''
do_replace("4-init-signature", OLD_INIT, NEW_INIT)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("Steps done so far:", steps_done)

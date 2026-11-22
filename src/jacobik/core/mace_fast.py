"""Sync-free, CUDA-graph-captured MACE-POLAR energy + force evaluation for the batched relaxer (opt-in).

Why: at the batch sizes of a relaxation, one MACE-POLAR-1-S evaluation on an A100 is latency-bound (about 50 ms for a
single 34-atom structure, most of it host-device syncs and kernel launches), and the active set of a relaxation shrinks
to a few slow rows that each cost a full latency per step. What this module changes, all exact up to float32 round-off:

- edges: every intra-structure atom pair instead of a per-step distance filter. MACE's radial cutoff envelope is zero
  beyond r_max, so a pair past the cutoff contributes nothing; the edge list is then fixed and needs no host sync.
- the long-range block (graph_longrange) runs in its "realspace" mode (what "auto" selects for a molecule) without the
  per-call torch.any(pbc); its complete charge graph, rebuilt by a Python loop over structures with one sync per
  structure on every call, is built once per batch; scatter sums get their output size from the batch instead of
  index.max(); a few CPU-index gathers (x[:, [0, 3, 1, 2]]) use device indices.
- the complete charge graph's pair sums (energy and field features) run as dense per-molecule blocks instead of
  gathers and scatters over its (4n)^2 edges per molecule: the index kernels, and the sorts deterministic mode adds to
  them, were the largest GPU cost of a settle pass (2026-11-21, RTX 2060).
- with those syncs gone the forward + force pass is captured as one CUDA graph per batch membership and replayed.

Patches are installed process-wide on first use but only change behaviour inside batch_context (outside it every
patched function runs the library's own code). Measured 2026-11-19, A100 login node (shared): one 34-atom structure
133.7 ms (relax._ef) -> 51 ms (full pairs) -> 9.3 ms (graph replay); 16 x 34 atoms 160 -> 56 ms. dF <= 3e-6 eV/A.
"""
from __future__ import annotations

import contextlib
import math
from typing import Dict, Optional

import torch

_CTX: Dict[str, object] = {"n_graphs": None, "n_nodes": None, "bcg": None}
_IDX: Dict[tuple, torch.Tensor] = {}
_INSTALLED = False


def _dev_index(vals, device):
    k = (tuple(vals), device)
    t = _IDX.get(k)
    if t is None:
        t = _IDX[k] = torch.tensor(list(vals), dtype=torch.long, device=device)
    return t


def _dense_layout(batch: torch.Tensor, N: int, n_graphs: int):
    """The complete charge graph of batch_complete_graph_excluding_self_duplicates_vector(batch, N) as padded blocks:
    idx (G, L) extended-node index per molecule, valid (G, L), pair mask (G, L, L) (same molecule, different original
    atom). The extended nodes are batch.repeat_interleave(N), so molecule g's nodes are contiguous when batch is
    sorted (PyG collation); None otherwise (the library's edge-list code then runs). One host sync, once per batch."""
    b = batch.long()
    if b.numel() > 1 and not bool((b[1:] >= b[:-1]).all()):
        return None
    counts = torch.bincount(b, minlength=n_graphs) * N
    L = int(counts.max())
    start = torch.cumsum(counts, 0) - counts
    ar = torch.arange(L, device=b.device)
    valid = ar[None, :] < counts[:, None]
    idx = torch.where(valid, start[:, None] + ar[None, :], torch.zeros((), dtype=torch.long, device=b.device))
    orig = idx // N
    M = valid[:, :, None] & valid[:, None, :] & (orig[:, :, None] != orig[:, None, :])
    sel = valid.reshape(-1).nonzero()[:, 0]          # padded (G * L) -> node order, as an index (capturable)
    return idx, valid, M, sel


def install():
    """Install the patches once (idempotent)."""
    global _INSTALLED
    if _INSTALLED:
        return
    import graph_longrange.energy as GE
    import graph_longrange.features as GF
    import graph_longrange.gto_utils as GU
    import graph_longrange.realspace_electrostatics as RSE
    import mace.modules.extensions as MX
    from mace.tools.scatter import scatter_sum as _ss

    def active():
        return _CTX["n_graphs"] is not None

    perm0 = MX._permute_to_e3nn_convention

    def permute(x):
        if not active():
            return perm0(x)
        return x.index_select(-1, _dev_index((1, 2, 0), x.device))
    MX._permute_to_e3nn_convention = permute

    bcg0 = RSE.batch_complete_graph_excluding_self_duplicates_vector

    def bcg(batch, N):
        cache = _CTX["bcg"]
        if not active() or cache is None:
            return bcg0(batch, N)
        k = (batch.data_ptr(), batch.numel(), int(N))
        if k not in cache:
            e = cache[k] = bcg0(batch, N)
            cache[("dense", e.data_ptr(), e.shape[1])] = _dense_layout(batch, int(N), _CTX["n_graphs"])
        return cache[k]
    RSE.batch_complete_graph_excluding_self_duplicates_vector = bcg

    from graph_longrange.utils import FIELD_CONSTANT
    ce0, cf0 = RSE.charges_energy_from_graph, RSE.charges_features_from_graph
    k4pi = FIELD_CONSTANT / (4 * math.pi)

    def layout(edge_index):
        cache = _CTX["bcg"]
        return None if (not active() or cache is None) else cache.get(("dense", edge_index.data_ptr(), edge_index.shape[1]))

    def dense(lay, charges, positions):
        """per-molecule padded charges (G, L) and pair distances (G, L, L), masked pairs at distance 1"""
        idx, valid, M, _ = lay
        P = positions[idx]
        diff = P[:, :, None, :] - P[:, None, :, :]
        d = torch.sqrt(torch.where(M, (diff * diff).sum(-1), torch.ones((), dtype=P.dtype, device=P.device)))
        return charges[idx] * valid, d

    def ce(charges, positions, edge_index, batch, density_smearing_width):
        lay = layout(edge_index)
        if lay is None:
            return ce0(charges, positions, edge_index, batch, density_smearing_width)
        q, d = dense(lay, charges, positions)
        s = torch.erf(d * 0.5 / density_smearing_width) / (d.abs() + 1e-6) * lay[2]
        return 0.5 * k4pi * torch.einsum("gi,gij,gj->g", q, s, q)
    RSE.charges_energy_from_graph = ce

    def cf(charges, positions, edge_index, batch, total_width_factors):
        lay = layout(edge_index)
        if lay is None:
            return cf0(charges, positions, edge_index, batch, total_width_factors)
        q, d = dense(lay, charges, positions)
        d = d[..., None]
        s = torch.erf(0.5 * d / total_width_factors) / (d + 1e-6) * lay[2][..., None]      # (G, L, L, R)
        f = torch.einsum("gj,gijr->gir", q, s)
        return k4pi * f.reshape(-1, f.shape[-1]).index_select(0, lay[3])                  # receivers in node order
    RSE.charges_features_from_graph = cf

    ext0 = GU.DisplacedGTOExternalFieldBlock.forward

    def ext(self, batch, positions, field):
        if not active():
            return ext0(self, batch, positions, field)
        nf = torch.index_select(field, 0, batch)
        pfd = torch.einsum("bi,bi->b", positions, nf[:, 1:].clone())
        nf = nf.clone()
        nf[:, 0] += pfd
        return torch.einsum("pf,nf->np", self.matrix, nf.index_select(1, _dev_index((0, 3, 1, 2), nf.device)))
    GU.DisplacedGTOExternalFieldBlock.forward = ext

    int0 = GU.GTOInternalFieldtoFeaturesBlock.forward

    def intf(self, batch, positions, node_fields):
        if not active():
            return int0(self, batch, positions, node_fields)
        return torch.einsum("pf,nf->np", self.matrix,
                            node_fields.index_select(1, _dev_index((0, 3, 1, 2), node_fields.device)))
    GU.GTOInternalFieldtoFeaturesBlock.forward = intf

    tcd0 = MX.compute_total_charge_dipole_permuted

    def tcd(density_coefficients, positions, batch, num_graphs):
        if not active():
            return tcd0(density_coefficients, positions, batch, num_graphs)
        dip = _ss(src=positions * density_coefficients[:, :1], index=batch.unsqueeze(-1), dim=0, dim_size=num_graphs)
        if density_coefficients.shape[1] > 1:
            dp = _ss(src=density_coefficients[..., 1:4], index=batch, dim=-2, dim_size=num_graphs)
            dip = dip + dp.index_select(-1, _dev_index((2, 0, 1), dp.device))
        return _ss(src=density_coefficients[:, 0], index=batch, dim=-1, dim_size=num_graphs), dip
    MX.compute_total_charge_dipole_permuted = tcd

    def ss(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
        if dim_size is None and active():
            n = _CTX["n_nodes"]
            if index.dim() == 1 and index.numel() in (n, 4 * n):       # the batch vector (or its 4x duplicate)
                dim_size = _CTX["n_graphs"]
            else:
                raise RuntimeError(f"mace_fast: scatter without dim_size over {index.numel()} entries")
        return _ss(src=src, index=index, dim=dim, out=out, dim_size=dim_size, reduce=reduce)
    for mod in (RSE, GE, GF, GU):
        if hasattr(mod, "scatter_sum"):
            mod.scatter_sum = ss
    _INSTALLED = True


@contextlib.contextmanager
def batch_context(model, n_graphs: int, n_nodes: int, bcg_cache: Optional[dict] = None):
    """Inside: patched code paths for a batch of n_graphs molecules with n_nodes atoms (no pbc). bcg_cache keeps the
    complete charge graphs of the batch alive (a captured graph reads them on every replay)."""
    install()
    prev = getattr(model, "pbc_handling", "auto")
    model.set_electrostatic_pbcs("realspace")
    if hasattr(model, "pbc_handling"):
        model.pbc_handling = "realspace"
    _CTX.update(n_graphs=int(n_graphs), n_nodes=int(n_nodes), bcg=bcg_cache)
    try:
        yield
    finally:
        _CTX.update(n_graphs=None, n_nodes=None, bcg=None)
        model.set_electrostatic_pbcs(prev)
        if hasattr(model, "pbc_handling"):
            model.pbc_handling = prev


class GraphEvaluator:
    """Energy (G,) and forces (N, 3) of one fixed collated batch (relax._collate / _subset output, all intra-structure
    pairs in d["_pairs"]), captured as a CUDA graph on the first call; eager full-pair fallback on any capture failure
    (self.mode says which)."""

    def __init__(self, model, d, mdt, energy_unit: float, graph: bool = True, X=None, r_list: Optional[float] = None,
                 r_cut: Optional[float] = None):
        """X, r_list: keep only the pairs within r_list (A) at positions X (flat, N x 3); the list holds while every
        pair left out stays beyond the model cutoff r_cut (stale() tests exactly that)."""
        self.model, self.mdt, self.u = model, mdt, energy_unit
        pi, pj = d["_pairs"]
        self.out, self.r_cut = None, r_cut
        if X is not None and r_list is not None:
            with torch.no_grad():
                keep = ((X[pi] - X[pj]) ** 2).sum(-1) < r_list ** 2
            self.out = (pi[~keep], pj[~keep])
            pi, pj = pi[keep], pj[keep]
        self.n_graphs = int(d["ptr"].numel() - 1)
        self.n_nodes = int(d["batch"].numel())
        inp = {k: v for k, v in d.items() if not k.startswith("_")}
        inp["edge_index"] = torch.stack([pi, pj])
        z = torch.zeros(pi.numel(), 3, dtype=mdt, device=pi.device)
        inp["shifts"], inp["unit_shifts"] = z, z
        self.inp = inp
        self.bcg: dict = {}
        self.g = None
        self.mode = "eager"
        self.want_graph = graph and pi.is_cuda

    def stale(self, X) -> bool:
        """True when an atom has moved far enough from the list positions that a pair outside the list may be inside
        the model cutoff."""
        if self.out is None or self.out[0].numel() == 0:
            return False
        oi, oj = self.out
        with torch.no_grad():
            return bool(((X[oi] - X[oj]) ** 2).sum(-1).min() < self.r_cut ** 2)

    def _run(self):
        out = self.model(self.inp, compute_force=True, compute_virials=False, compute_stress=False, training=False)
        return self._energy64(out), out["forces"]

    def _energy64(self, out):
        """The total energy summed in float64: per-atom energies (atomic reference + local terms, float64 in the
        model's output) plus the long-range electrostatic and local electron terms. The model's own float32 total of
        ~6e4 eV is quantised to 3.9 meV and its rounding moves by several ulps with the batch membership and along a
        path, more than an optimiser's energy decrease near convergence (measured: membership shift 11.7 meV -> 7e-7
        meV; the float32 total sits 34-41 meV above this sum)."""
        from mace.tools.scatter import scatter_sum

        ne, es = out.get("node_energy"), out.get("electrostatic_energy")
        if ne is None or es is None:
            return out["energy"].double()
        e = scatter_sum(ne.double(), self.inp["batch"], dim=-1, dim_size=self.n_graphs) + es.double()
        if getattr(self.model, "add_local_electron_energy", False) and out.get("electron_energy") is not None:
            e = e + out["electron_energy"].double()
        return e

    def _capture(self, X):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self.inp["positions"] = X.to(self.mdt).detach().clone()
            self._run()                                                   # warm-up (allocations, lazy init)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            self.e, self.f = self._run()
        self.g = g
        self.mode = "graph"

    def __call__(self, X):
        with batch_context(self.model, self.n_graphs, self.n_nodes, self.bcg):
            if self.want_graph and self.g is None:
                try:
                    self._capture(X)
                except Exception as exc:                                  # degrade to the eager full-pair path
                    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc):
                        raise                                             # the relaxer splits the chunk
                    self.want_graph, self.g, self.mode = False, None, f"eager ({type(exc).__name__}: {exc})"[:200]
                    torch.cuda.synchronize()
            if self.g is not None:
                with torch.no_grad():
                    self.inp["positions"].copy_(X.to(self.mdt))
                self.g.replay()
                e, f = self.e, self.f
            else:
                self.inp["positions"] = X.to(self.mdt).detach().clone()
                e, f = self._run()
        e = e.detach().double() * self.u
        return e, e, f.detach().double() * self.u

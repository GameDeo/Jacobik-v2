"""S7: donor parsing against the SMILES (research/archive/docs/roadmap/roadmap_generation_v1.md S7). CPU only.

For each listed parent: the dative arrows in the tmQM SMILES ("->" and "<-" tokens, with ring-closure digits noted)
against the number of donor labels the solver's shell gets (len(shell.donor_labels), the classifier output). Haptic
parents are reported but not classed (arrow counts are meaningless there). Reads SMILES and geometry only.

    python tools/jgen/diag/d32_s7_donor_parse.py [CODE ...]
"""
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), ROOT, os.path.join(ROOT, "refit"), os.path.join(ROOT, "tools", "jdev", "budget")]

LOST = ["IYUKOC", "OREHAV", "VAVBIE", "NEPNEF", "VOBGUR", "JAYCEV", "BUXJIO", "PEMDAP", "EHUBUH", "OBABAX", "RUTVAD",
        "TINNIO", "TAYJIQ", "NASVAI"]
EXTRA = ["UMUBAF", "MIDQEY", "BOHWUT", "NUHPEL", "WAYDIN", "WIMGIL", "RAYSOZ", "LUQWOK", "YUCZUS", "SOBHAS", "VIFRIL"]


def arrows(smi):
    toks = re.findall(r"(->|<-)(\d?)", smi or "")
    return dict(n=len(toks), ring_closure=sum(1 for _, d in toks if d), forward=sum(1 for t, _ in toks if t == "->"),
                backward=sum(1 for t, _ in toks if t == "<-"))


def main():
    from tools.jgen import inputs
    from jacobik.core.env import build_builder_and_inputs

    codes = sys.argv[1:] or (LOST + EXTRA)
    smiles = inputs.tmqm_smiles(inputs.CLEAN / "tmQM")
    shards = inputs.frame_shards()
    rows = []
    for code in codes:
        rec = dict(code=code, expected="lost" if code in LOST else ("extra" if code in EXTRA else "other"))
        smi = smiles.get(code)
        rec["arrows"] = arrows(smi)
        if code not in shards:
            rec["error"] = "not in frame"
            rows.append(rec)
            continue
        shard, ds = shards[code]
        mol = inputs.load_mol(code, shard, ds)
        b, _, err = build_builder_and_inputs(mol)
        if b is None:
            rec["error"] = f"unmappable: {err}"
            rows.append(rec)
            continue
        shell, _, _ = b._build_shell(False, False, chelate_embed=True)
        n_lab = len(getattr(shell, "donor_labels", {}) or {})
        n_site = len(shell.coord_sites)
        haptic = any(getattr(s, "is_haptic", False) for s in shell.coord_sites)
        rec.update(donor_labels=n_lab, coord_sites=n_site, haptic=haptic, geometry=getattr(shell, "coord_geom", None) and
                   getattr(shell.coord_geom, "name", None))
        rec["arrows_minus_donors"] = rec["arrows"]["n"] - n_lab
        rows.append(rec)
    print(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()

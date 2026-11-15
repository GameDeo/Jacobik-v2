import json

import pytest

from tools.runkit import policy
from tools.runkit.trope import cover, suite, tags


def feat(**kw):
    f = dict(n_atoms=30, metal="Pd", charge=0, cn=4, coord_geom="square_plan", donor_subclasses=[], ligands=[])
    f.update(kw)
    return f


def lig(**kw):
    d = dict(n_atoms=10, donors=[], subclasses=[], haptic_order=0, haptic_donors=[], macrocycle=False, rings=[], phosphine=[], carbene=[])
    d.update(kw)
    return d


def test_tag_rules():
    f = feat(ligands=[lig(rings=[5, 9]), lig(haptic_order=5, haptic_donors=["C"] * 5), lig(haptic_order=2, haptic_donors=["N", "N"]),
                      lig(macrocycle=True), lig(phosphine=[dict(n_c=3, branching=[3, 3, 3])]),
                      lig(carbene=[dict(n_N=2, n_H=0, n_nb=2)]), lig(subclasses=["C_alkylidene"])],
             donor_subclasses=["Hal_Cl", "N_amido", "O_phenoxide"])
    t = tags.tags_of(f)
    assert {"chelate_ring5", "chelate_ring8plus", "haptic_eta5", "macrocycle", "bulky_phosphine", "carbene_nhc", "carbene_other",
            "anionic_halide", "anionic_N", "anionic_O", "cn4_square_plan"} <= t
    assert "haptic_eta2" not in t                                   # N,N "eta-2" is not a haptic carbon ligand
    assert "bulky_phosphine" not in tags.tags_of(feat(ligands=[lig(phosphine=[dict(n_c=3, branching=[0, 0, 0])])]))
    assert "lanthanide" in tags.tags_of(feat(metal="La"))
    assert "cn5_square_pyr" in tags.tags_of(feat(cn=5, coord_geom="square_pyr"))


def test_cover_full_and_deterministic():
    P = [dict(code="A", n_atoms=20, tags=["x", "y"]), dict(code="B", n_atoms=20, tags=["y", "z"]), dict(code="C", n_atoms=20, tags=["w"]),
         dict(code="D", n_atoms=500, tags=["q"])]                    # D too large to be eligible
    ch, unc, freq, par = cover.best_cover(P)
    assert unc == [] or unc == ["q"] or set(unc) <= {"q"}
    assert {t for c in ch for t in c["all_tags"]} >= {"x", "y", "z", "w"}
    assert cover.best_cover(P)[0] == ch


def test_committed_suite_is_clean():
    d = json.loads((suite.Path(suite.__file__).resolve().parents[5] / "research/archive/code/trope_suite/suite.json").read_text())
    pol = policy.DataPolicy()
    codes = [p["code"] for p in d["parents"]]
    pol.assert_allowed(codes)                                       # no val, no reserve
    assert not ({"SOBHAS", "YUCZUS"} & set(codes))
    assert 8 <= len(codes) <= 12 and d["uncovered_tags"] == []


def rec(code, host="h", digest="a", score=1.0, dE=10.0, strain=5.0):
    b = [dict(embed_idx=0, assignment_idx=i, is_top1=(i == 0), score=score - i, dE_unrel=dE + i, strain=strain + i, offset=2.0 * i)
         for i in range(3)]
    return dict(code=code, status="ok", reason="", host=host, fingerprint="f", per_seed={"11": dict(
        status="ok", geom_digest=digest, scores_digest=digest, top1_score=score, dE_unrel=dE, strain=strain, offset=0.0,
        md_bond_dev=1.0, rmsd_ref=0.5, n_buckets=3, buckets=b)})


def test_compare_flags_violation_outside_footprint(disk_tmp):
    s = dict(parents=[dict(code="IN", all_tags=["macrocycle"]), dict(code="OUT", all_tags=["cn4_tetrahedral"])])
    c, a = disk_tmp / "c", disk_tmp / "a"
    c.mkdir(); a.mkdir()
    for code in ("IN", "OUT"):
        (c / f"{code}.json").write_text(json.dumps(rec(code)))
    (a / "IN.json").write_text(json.dumps(rec("IN", digest="b", score=1.5, dE=4.0)))
    (a / "OUT.json").write_text(json.dumps(rec("OUT")))
    foot = dict(tags=["macrocycle"], parents=[])
    r = suite.compare(c, a, foot, suite=s)
    assert r["outside_ok"] and r["per_trope"]["macrocycle"]["d_dE_unrel"] == -6.0
    assert r["per_trope"]["macrocycle"]["d_bucket_strain"] == 0.0 and len(r["rows"][0]["buckets"]) == 3
    (a / "OUT.json").write_text(json.dumps(rec("OUT", digest="zzz")))
    r = suite.compare(c, a, foot, suite=s)
    assert not r["outside_ok"] and r["outside_violations"] == ["OUT"]
    (a / "OUT.json").write_text(json.dumps(rec("OUT", host="other")))
    assert suite.compare(c, a, foot, suite=s)["rows"][1]["status"] == "unpaired"


def test_seeds_only_inside_footprint():
    foot = dict(tags=["macrocycle"], parents=["P"])
    assert suite.seeds_for("X", ["macrocycle"], foot) == ((11, 12, 13), True)
    assert suite.seeds_for("X", ["lanthanide"], foot) == ((11,), False)
    assert suite.seeds_for("P", [], foot)[1]


def test_md_bond_dev_zero_for_reference():
    import numpy as np
    X = np.array([[0, 0, 0], [2.0, 0, 0], [0, 1.0, 0]], float)
    adj = np.zeros((3, 3), bool); adj[1, 2] = adj[2, 1] = True
    assert suite.md_bond_dev(["M", "N", "H"], X, X, 0, adj, [1]) == 0.0
    Y = X.copy(); Y[1, 0] = 2.1
    assert suite.md_bond_dev(["M", "N", "H"], X, Y, 0, adj, [1]) > 0

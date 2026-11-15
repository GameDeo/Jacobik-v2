"""Source-only donor reconciliation census; no arm energy or generated structures."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

T1D = {"hapto_bond_gate": True, "hapto_label_canon": True, "hapto_chelate_exclusion": True}
MODES = {"C": {}, "D": T1D,
         # DS_CENSUS_DR_EXTRA (JSON) adds flags to the reconcile arm, e.g. {"donor_assignment": true}
         "DR": {**T1D, "donor_identity_reconcile": True, **json.loads(os.environ.get("DS_CENSUS_DR_EXTRA", "{}"))}}


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    Path(p).write_text(json.dumps(v, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare(main, out):
    """Freeze exactly the watched training sources and previously audited repair fixtures."""
    from fit.parser import _iter_xyz_blocks, _iter_bo_blocks, _parse_xyz_comment
    evidence = main / "research/data/audit/2026-11-10_DS"
    watched = {p["code"]: p for p in read(evidence / "q1_training_manifest.json")}
    sources, expected = {}, {}
    old_manifest = {p["code"]: p for p in read(evidence / "gate_inputs/manifest.json")["parents"]}
    for fixture in read(evidence / "gate_inputs/target_overlays_v1.json")["overlays"]:
        code = fixture["code"]
        source = read(old_manifest[code]["source_file"])
        if digest(source) != fixture["source_sha256"]:
            raise ValueError("source does not match frozen corrected fixture")
        sources[code] = source
        expected[code] = sorted(s["atoms"] for s in fixture["corrected_sites_xyz"])
    # A verified unchanged local negative control, not another optimisation case.
    sources["LUQVOK"] = read(old_manifest["LUQVOK"]["source_file"])
    by_shard = defaultdict(set)
    for code, row in watched.items():
        if code not in sources:
            by_shard[row["shard"]].add(code)
    for shard, wanted in sorted(by_shard.items()):
        source_root = main / "data/ref/tmQM"
        with (source_root / "tmQM_y.csv").open(newline="") as f:
            rows = csv.reader(f, delimiter=";"); head = next(rows)
            ci, si = head.index("CSD_code"), head.index("SMILES")
            smiles = {r[ci].strip(): r[si].strip() or None for r in rows if r and r[ci].strip() in wanted}
        xyz, bo = {}, {}
        for comment, lines in _iter_xyz_blocks(source_root / "xyz" / f"tmQM_{shard}.xyz"):
            code = _parse_xyz_comment(comment)[0]
            if code in wanted:
                if code in xyz: raise ValueError("duplicate source XYZ")
                xyz[code] = (comment, lines)
        for code, lines in _iter_bo_blocks(source_root / "bo" / f"tmQM_{shard}.BO"):
            if code in wanted:
                if code in bo: raise ValueError("duplicate source BO")
                bo[code] = lines
        for code in wanted:
            comment, lines = xyz[code]
            sources[code] = dict(code=code, smiles=smiles.get(code), xyz_comment=comment,
                                 xyz_atoms=lines, bo_lines=bo[code], dataset="tmQM", shard=shard)
    dest = out / "source_blocks"; dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for code, source in sorted(sources.items()):
        path = dest / f"{code}.json"; write(path, source)
        for mode in MODES:
            rows.append(dict(id=f"{code}_{mode}", code=code, mode=mode, source_file=str(path.resolve()),
                             source_sha256=digest(source), watched_training=code in watched,
                             expected_sites=expected.get(code)))
    write(out / "manifest.json", rows)
    write(out / "pilot_manifest.json", [r for r in rows if r["code"] in {"LUQVOK", "BOPSAD", "DUCPAT"}])
    return dict(parents=len(sources), rows=len(rows), training=len(watched), audited_fixtures=len(expected))


def run(parent):
    os.environ["JACOBIK_PROFILE"] = "production"
    os.environ["JACOBIK_PROFILE_OVERRIDES"] = json.dumps(MODES[parent["mode"]])
    import numpy as np
    from fit.parser import _assemble_raw
    from jacobik.topology.mapping import assemble, _covalent_radius
    from jacobik.stats.keys import q1_site_keys
    from fit.q1 import q1_bond_lengths
    source = read(parent["source_file"])
    if source["code"] != parent["code"] or digest(source) != parent["source_sha256"]:
        raise ValueError("source identity mismatch")
    raw = _assemble_raw(source["xyz_comment"], source["xyz_atoms"], source["code"], source["bo_lines"],
                        SimpleNamespace(smiles=source["smiles"]))
    atom_map = {a.idx: a for a in raw.atoms}
    raw_bonds = {(min(b.i, b.j), max(b.i, b.j)): b.order for b in raw.bonds}
    mol = assemble(raw)
    sites = []
    for site, key in zip(mol.coord_sites, q1_site_keys(mol), strict=True):
        contacts = []
        for i in site.donor_indices:
            atom = atom_map[i]
            dist = float(np.linalg.norm(atom.position - atom_map[mol.metal_idx].position))
            contacts.append(dict(atom=i, element=atom.symbol, distance_A=dist,
                                 ratio=dist / (_covalent_radius(mol.metal_symbol) + _covalent_radius(atom.symbol)),
                                 metal_bo=raw_bonds.get((min(i,mol.metal_idx),max(i,mol.metal_idx)),0.0)))
        sites.append(dict(atoms=sorted(site.donor_indices), eta=site.haptic_order, key=key, contacts=contacts))
    actual = sorted(s["atoms"] for s in sites)
    return dict(code=parent["code"], mode=parent["mode"], source_sha256=parent["source_sha256"],
                watched_training=parent["watched_training"], geometry=mol.coord_geom, sites=sites,
                expected_sites=parent["expected_sites"],
                matches_audited_target=(actual == parent["expected_sites"] if parent["expected_sites"] is not None else None),
                events=getattr(mol,"donor_identity_events",[]), q1_observations=q1_bond_lengths(mol))


def review_taxonomy(record):
    """Describe uncertainty without converting a diagnostic into an exclusion."""
    sites = record["sites"]
    contacts = {c["atom"]: c for site in sites for c in site["contacts"]}
    haptic = {i for site in sites if site["eta"] >= 2 for i in site["atoms"]}
    groups = defaultdict(set)
    for event in record["events"]:
        if event["kind"] != "weak_declared_identity_retained":
            continue
        for i in event["atoms"]:
            if i not in contacts:
                category = "not_in_final_q1_sites"
            elif i in haptic:
                category = "retained_haptic_constituent"
            elif contacts[i]["metal_bo"] >= .15 and contacts[i]["ratio"] <= 1.31:
                category = "boundary_declared_kappa_review"
            else:
                category = "unresolved_primary_identity_review"
            groups[category].add(i)
    hydrogen_ambiguous = any(e["kind"] == "hydrogen_alignment_ambiguous" for e in record["events"])
    review_atoms = groups["unresolved_primary_identity_review"] | groups["boundary_declared_kappa_review"]
    prior_collision = any(e["kind"] == "legacy_supported_identity_collision" for e in record["events"])
    affected_keys = [s["key"] for s in sites if hydrogen_ambiguous or prior_collision or record["matches_audited_target"] is False
                     or review_atoms.intersection(s["atoms"])]
    return dict(weak_atom_categories={k:sorted(v) for k,v in groups.items() if v},
                hydrogen_alignment_ambiguous=hydrogen_ambiguous, prior_correspondence_collision=prior_collision,
                audited_target_disagrees=record["matches_audited_target"] is False,
                q1_keys_requiring_source_review=affected_keys,
                quarantine_applied=False)


def summarize(out, evidence):
    manifest = read(out / "manifest.json")
    by_code = defaultdict(dict)
    for row in manifest:
        got = read(out / "run_v1/records" / f"{row['id']}.json")
        if got.get("status") != "ok": raise ValueError(f"failed source task {row['id']}")
        by_code[row["code"]][row["mode"]] = got
    ledger = []
    for code, modes in sorted(by_code.items()):
        old, new = modes["D"], modes["DR"]
        old_sites = sorted(s["atoms"] for s in old["sites"])
        new_sites = sorted(s["atoms"] for s in new["sites"])
        expected_atoms = {i for site in new["expected_sites"] or [] for i in site}
        old_atoms = {i for site in old_sites for i in site}
        new_atoms = {i for site in new_sites for i in site}
        ledger.append(dict(code=code, source_sha256=new["source_sha256"], watched_training=new["watched_training"],
                           old_sites=old_sites, new_sites=new_sites, changed=old_sites != new_sites,
                           old_geometry=old["geometry"], new_geometry=new["geometry"], events=new["events"],
                           matches_audited_target=new["matches_audited_target"],
                           lost_expected_atoms=sorted((old_atoms & expected_atoms) - new_atoms),
                           newly_added_nonexpected_atoms=(sorted((new_atoms - old_atoms) - expected_atoms)
                                                          if new["expected_sites"] is not None else None),
                           before_q1=old["q1_observations"],after_q1=new["q1_observations"],
                           source_review=review_taxonomy(new)))
    # Read only retained observation rows for the 11 prespecified training cases.
    import pyarrow.parquet as pq
    watched = [c for c, m in by_code.items() if m["DR"]["watched_training"]]
    overview = []
    for split in ("X1","X2","X3","LNf"):
        path=evidence / "fit_inputs/d2_obs" / f"observations_Q1_{split}.parquet"
        table=pq.read_table(path,columns=["csd_code","key","value"],filters=[("csd_code","in",watched)])
        overview.extend(table.to_pylist())
    summary=dict(parents=len(by_code),source_tasks=len(manifest),
                 training_parents=len(watched), changed=sum(r["changed"] for r in ledger),
                 audited_target_matches=sum(r["matches_audited_target"] is True for r in ledger),
                 audited_target_mismatches=[r["code"] for r in ledger if r["matches_audited_target"] is False],
                 event_counts=dict(Counter(e["kind"] for r in ledger for e in r["events"])),
                 retained_q1_rows=len(overview),retained_training_parent_count=len({r["csd_code"] for r in overview}),
                 review_category_counts=dict(Counter(k for r in ledger for k in r["source_review"]["weak_atom_categories"])),
                 hydrogen_alignment_ambiguous=[r["code"] for r in ledger if r["source_review"]["hydrogen_alignment_ambiguous"]])
    write(out/"correction_ledger.json",ledger);write(out/"retained_q1_watched_rows.json",overview)
    write(out/"census_summary.json",summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=["prepare","summarize"])
    p.add_argument("--main",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    result=prepare(a.main,a.out) if a.action=="prepare" else summarize(a.out,a.main/"research/data/audit/2026-11-10_DS")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":main()

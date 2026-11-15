"""Follow-up chemotype screens for H-2 audit: eta2 delocalized anions and N/C planarity candidates."""
from __future__ import annotations
import json, glob, sys
from collections import defaultdict
from pathlib import Path
sys.path[:0] = ["src", ".", "research/bench"]
from rdkit import Chem
from tools.jgen import inputs
from jacobik.core.env import build_builder_and_inputs
from jacobik.builder.embed import undecided_ez

OUT=Path("research/data/jgen/campaign/H/h2_census/chemotype_refinements.json")
CENSUS=Path("research/data/jgen/campaign/H/h2_census")

def components(mol):
    adj={i:set() for i in range(mol.GetNumAtoms())}
    for b in mol.GetBonds():
        if b.GetIsConjugated():
            i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx();adj[i].add(j);adj[j].add(i)
    seen=set();out=[]
    for i in adj:
        if i in seen: continue
        todo=[i];seen.add(i);cc=[]
        while todo:
            u=todo.pop();cc.append(u)
            for v in adj[u]-seen: seen.add(v);todo.append(v)
        out.append(set(cc))
    return out

def ring_sets(mol):
    # RDKit's perceived atom rings provide enough local context for this screen.
    aromatic=set()
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) not in (5,6): continue
        atoms=[mol.GetAtomWithIdx(i) for i in ring]
        bonds=[mol.GetBondBetweenAtoms(ring[k],ring[(k+1)%len(ring)]) for k in range(len(ring))]
        if all(b is not None and (b.GetIsAromatic() or b.GetBondTypeAsDouble()>=1.4) for b in bonds):
            aromatic.update(ring)
    return aromatic

def main():
    # Get unique parents with a recorded anionic two-atom donor group.
    candidate_codes=set()
    for p in list(CENSUS.glob("*_cal.json"))+list(CENSUS.glob("*_tail.json")):
        d=json.loads(p.read_text())
        if any(len(x.get("group",[]))==2 and x.get("net_formal_charge",0)<0 for x in d.get("haptic_anions",[])):
            candidate_codes.add(d["code"])
    index=inputs.frame_shards(); eta2=[]
    for code in sorted(candidate_codes):
        sh,ds=index[code];ref=inputs.load_mol(code,sh,ds);builder,_,err=build_builder_and_inputs(ref)
        if builder is None: raise RuntimeError(f"{code}: {err}")
        for li,lig in enumerate(builder.ligands):
            m=Chem.MolFromSmiles(lig.smiles)
            if m is None: continue
            ccs=components(m)
            for group in lig.donors:
                if len(group)!=2 or sum(m.GetAtomWithIdx(int(i)).GetFormalCharge() for i in group)>=0: continue
                g=set(map(int,group)); hits=[]
                for cc in ccs:
                    charge=sum(m.GetAtomWithIdx(i).GetFormalCharge() for i in cc)
                    if g & cc and charge<0 and len(cc)>=3: hits.append(cc)
                for cc in hits:
                    ez=[x for x in undecided_ez(m,lig.donors) if x[0] in cc and x[1] in cc]
                    eta2.append({"code":code,"ligand":li,"donor_group":sorted(g),"group_charge":-1,
                        "conjugated_component":sorted(cc),"component_charge":sum(m.GetAtomWithIdx(i).GetFormalCharge() for i in cc),
                        "unresolved_ez_bonds_in_component":[list(x[:2]) for x in ez]})

    # Reclassify the existing reference-planar / embed-pyramidal candidates by graph environment.
    angle_candidates=defaultdict(list)
    for p in CENSUS.glob("*_cal.json"):
        if p.name.startswith("rings_"): continue
        d=json.loads(p.read_text())
        angle_candidates[d["code"]]=[x for x in d.get("angle_sums",[])
            if x.get("reference_sum_deg",0)>=345 and x.get("embed_sum_deg") is not None and x["embed_sum_deg"]<345]
    types=defaultdict(set); centers=defaultdict(list)
    for code,items in angle_candidates.items():
        if not items: continue
        sh,ds=index[code];ref=inputs.load_mol(code,sh,ds)
        adj=[[] for _ in ref.atoms];bondmap={}
        for b in ref.bonds:
            i,j=int(b.i)-1,int(b.j)-1
            if 0<=i<len(adj) and 0<=j<len(adj): adj[i].append(j);adj[j].append(i);bondmap[frozenset((i,j))]=float(b.order)
        rings=[]
        # Simple graph cycle basis; classify aromatic five/six-member rings by edge orders.
        import networkx as nx
        gph=nx.Graph();gph.add_nodes_from(range(len(ref.atoms)))
        for pair,order in bondmap.items():
            if len(pair)==2:gph.add_edge(*tuple(pair),order=order)
        for cyc in nx.cycle_basis(gph):
            if len(cyc) not in (5,6):continue
            orders=[bondmap.get(frozenset((cyc[k],cyc[(k+1)%len(cyc)])),0) for k in range(len(cyc))]
            if sum(o>=1.4 for o in orders)>=2:rings.append(set(cyc))
        aromatic_atoms=set().union(*rings) if rings else set()
        for x in items:
            i=int(x["atom"]);el=ref.atoms[i].symbol;ns=[j for j in adj[i] if ref.atoms[j].symbol!="H"]
            kind=f"other_{el}"
            if el=="N":
                amide=False
                for c in ns:
                    if ref.atoms[c].symbol!="C":continue
                    if any(ref.atoms[o].symbol in ("O","S") and bondmap.get(frozenset((c,o)),0)>=1.8 for o in adj[c]):
                        amide=True;break
                if amide:kind="amide_N"
                elif i in aromatic_atoms:kind="ring_sp2_N"
                elif any(j in aromatic_atoms and ref.atoms[j].symbol=="C" for j in ns):kind="aniline_N"
                else:kind="other_N"
            elif el=="C":
                if any(ref.atoms[o].symbol in ("O","S") and bondmap.get(frozenset((i,o)),0)>=1.8 for o in adj[i]):kind="carbonyl_C"
                elif i in aromatic_atoms:kind="aromatic_C"
                else:kind="other_C"
            types[kind].add(code);centers[kind].append({"code":code,"atom":i,"element":el,
                "reference_sum_deg":x["reference_sum_deg"],"embed_sum_deg":x["embed_sum_deg"]})
    out={"eta2_delocalized_anionic_fragments":{"parent_count":len({x['code'] for x in eta2}),"group_count":len(eta2),"groups":eta2,
        "definition":"negative multi-atom donor group of size two intersects a conjugated RDKit bond component of at least three atoms with net formal charge <0"},
        "unresolved_ez_in_component_parent_count":len({x['code'] for x in eta2 if x['unresolved_ez_bonds_in_component']}),
        "planarity_candidates":{"total_parents":len({c for c,v in angle_candidates.items() if v}),
            "center_type_parent_count":{k:len(v) for k,v in sorted(types.items())},
            "center_type_center_count":{k:len(v) for k,v in sorted(centers.items())},
            "centers":dict(centers),
            "method":"heuristic graph classes from input atom/bond graph; candidate screen, not independent hybridization validation"}}
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=="__main__":main()

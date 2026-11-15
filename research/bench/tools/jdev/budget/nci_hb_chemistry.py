"""Experimental shared HB chemistry. No deployed imports/default changes.

Charge belongs to a local resonance group, not the entire ligand. Unknown
mapping/chemistry raises explicitly; it must never become a neutral fallback.
Training and generated graphs use the same RDKit descriptor and key ladder.
"""
from __future__ import annotations
from dataclasses import dataclass,replace
from collections import defaultdict
import json
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Lipinski

VERSION='hb_ctx1'


class ChemistryError(ValueError):pass


@dataclass(frozen=True)
class AtomContext:
    element:str
    family:str
    group_charge:int
    group_atoms:tuple[int,...]
    acceptor:bool
    donor:bool

    @property
    def charge(self):return 'anion' if self.group_charge<0 else 'cation' if self.group_charge>0 else 'neutral'


def _local_describe(rd):
    """RDKit indices -> context for a metal-free molecule, including explicit H.

    Lipinski acceptor chemistry is extended explicitly for water, neutral ammonia,
    anionic nitrogen and free halides. This is a conventional HB domain; covalent
    halogens and oxidized sulfur centers are excluded, not assigned weak wells.
    """
    # Lipinski's atom SMARTS assume implicit H; explicit H can otherwise make
    # an acid hydroxyl match the H0 acceptor alternative. Preserve heavy IDs.
    heavy=Chem.Mol(rd)
    for a in heavy.GetAtoms():a.SetIntProp('_hb_original_index',a.GetIdx())
    heavy=Chem.RemoveHs(heavy)
    acceptors={heavy.GetAtomWithIdx(i[0]).GetIntProp('_hb_original_index') for i in heavy.GetSubstructMatches(Lipinski.HAcceptorSmarts)}
    out={}
    for a in rd.GetAtoms():
        i=a.GetIdx();el=a.GetSymbol()
        if el=='H':continue
        neighbors=[n for n in a.GetNeighbors() if n.GetSymbol()!='H']
        nh=a.GetTotalNumHs(includeNeighbors=True);q=a.GetFormalCharge();group={i};family='other'
        donor=el in ('N','O','F') and nh>0
        acc=i in acceptors
        if el=='O':
            family='water' if not neighbors and nh==2 else 'hydroxide' if not neighbors and nh==1 and q<0 else 'oxide' if not neighbors and q==-2 else 'oxyl' if not neighbors and q==-1 else 'phenol' if nh and any(n.GetIsAromatic() for n in neighbors) else 'alcohol' if nh else 'alkoxide' if q<0 and len(neighbors)==1 and neighbors[0].GetSymbol()=='C' else 'ether'
            if not neighbors and (nh==2 or q<0):acc=True
            for center in neighbors:
                ce=center.GetSymbol();ox=[n for n in center.GetNeighbors() if n.GetSymbol()=='O']
                if ce=='C' and center.GetDegree()==1 and rd.GetBondBetweenAtoms(i,center.GetIdx()).GetBondTypeAsDouble()>=2.9:
                    # Neutral CO is often written [C-]#[O+]. Its oxygen is not
                    # a cationic acceptor group; weak CO interactions are outside
                    # this conventional HB prototype's admitted acceptor domain.
                    group={i,center.GetIdx()};q=sum(rd.GetAtomWithIdx(j).GetFormalCharge() for j in group);family='carbon_monoxide';acc=False;break
                has_double=any(rd.GetBondBetweenAtoms(center.GetIdx(),n.GetIdx()).GetBondTypeAsDouble()>=1.9 for n in ox)
                if (ce=='C' and len(ox)>=2 and has_double) or (ce in ('N','P','S') and len(ox)>=2):
                    group={center.GetIdx(),*(n.GetIdx() for n in ox)};q=sum(rd.GetAtomWithIdx(j).GetFormalCharge() for j in group)
                    if ce=='N' and len(ox)==2 and center.GetFormalCharge()==0 and all(n.GetFormalCharge()==-1 for n in ox) and not has_double:
                        raise ChemistryError('Noncanonical doubly negative N(O-)2 drawing needs explicit ionic normalization')
                    if ce=='C':family=('carbonate' if len(ox)>=3 else 'carboxylate') if q<0 else 'acid_hydroxyl' if nh else 'carboxyl_carbonyl' if rd.GetBondBetweenAtoms(i,center.GetIdx()).GetBondTypeAsDouble()>1.9 else 'ester_oxygen'
                    elif ce=='N':family='nitrate' if len(ox)==3 else 'nitro' if any(n.GetSymbol()=='C' for n in center.GetNeighbors()) else 'nitrite'
                    elif ce=='S':family='sulfonate' if any(n.GetSymbol()=='C' for n in center.GetNeighbors()) else 'sulfate'
                    else:family='phosphate'
                    break
                if rd.GetBondBetweenAtoms(i,center.GetIdx()).GetBondTypeAsDouble()>1.9:family='carbonyl' if ce=='C' else 'oxo'
        elif el=='N':
            if a.GetIsAromatic():family='pyrrole' if nh else 'pyridine'
            elif any(b.GetBondTypeAsDouble()>=2.9 for b in a.GetBonds()):family='nitrile'
            elif any(n.GetSymbol() in ('C','P','S') and any(b.GetOtherAtom(n).GetSymbol() in ('O','S') and b.GetBondTypeAsDouble()>=1.9 for b in n.GetBonds()) for n in neighbors):family='amide'
            elif any(b.GetBondTypeAsDouble()>=1.9 for b in a.GetBonds()):family='imine'
            else:family='ammonia' if not neighbors else 'aniline' if any(n.GetIsAromatic() for n in neighbors) else 'amine'
            if q<0:acc=True
            if not neighbors and nh==3 and q==0:acc=True
            if q>0:acc=False
        elif el=='S':
            oxidized=any(n.GetSymbol()=='O' and rd.GetBondBetweenAtoms(i,n.GetIdx()).GetBondTypeAsDouble()>1.9 for n in neighbors) or len(neighbors)>2
            family='oxidized_sulfur' if oxidized else 'thiolate' if q<0 else 'thiocarbonyl' if any(b.GetBondTypeAsDouble()>1.9 for b in a.GetBonds()) else 'thiol' if nh else 'thioether'
            if oxidized:acc=False
        elif el in ('F','Cl','Br','I'):
            family='halide' if q<0 and not neighbors and not nh else 'covalent_halogen'
            acc=family=='halide'
        else:acc=False
        out[i]=AtomContext(el,family,int(q),tuple(sorted(group)),bool(acc),bool(donor))
    return out


def describe(rd):
    """Extend local groups with charge delocalization across RDKit resonance.

    Enumerate each charged ligand fragment independently, with no charge-
    separation flag. A finite enumeration cap reports unsupported chemistry
    explicitly. No geometries or energy estimates enter group construction.
    """
    out={};fragment_ids=[]
    fragments=Chem.GetMolFrags(rd,asMols=True,sanitizeFrags=True,fragsMolAtomMapping=fragment_ids)
    for fragment,ids in zip(fragments,fragment_ids):
        # Reparse one canonical atom order before resonance enumeration. Direct
        # post-metal-removal caches/order can change RDKit's conjugation groups.
        # SMILES output order preserves an exact map back to the caller's atoms.
        smi=Chem.MolToSmiles(fragment,canonical=True)
        order=json.loads(fragment.GetProp('_smilesAtomOutputOrder'))
        params=Chem.SmilesParserParams();params.removeHs=False
        canonical=Chem.MolFromSmiles(smi,params)
        if canonical is None or canonical.GetNumAtoms()!=len(order):raise ChemistryError('Canonical fragment roundtrip lost atom identities')
        ids=tuple(ids[j] for j in order);fragment=canonical
        local=_local_describe(fragment);out.update(_remap(local,{i:ids[i] for i in range(len(ids))}))
        if not any(a.GetFormalCharge() for a in fragment.GetAtoms()):continue
        if not any(out.get(i) and (out[i].acceptor or out[i].donor) for i in ids):continue
        variants=list(Chem.ResonanceMolSupplier(fragment,maxStructs=65))
        if len(variants)>=65:raise ChemistryError('Charged ligand exceeds 64-form resonance enumeration budget')
        if len(variants)<2:continue
        graph=nx.Graph();varying=set()
        for variant in variants:
            for a in variant.GetAtoms():
                i=a.GetIdx()
                if a.GetFormalCharge()!=fragment.GetAtomWithIdx(i).GetFormalCharge():varying.add(i);graph.add_node(i)
            for b in variant.GetBonds():
                i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx();original=fragment.GetBondBetweenAtoms(i,j)
                if original is None:raise ChemistryError('Resonance changed molecular connectivity')
                if abs(b.GetBondTypeAsDouble()-original.GetBondTypeAsDouble())>1e-6:graph.add_edge(i,j)
        for group in nx.connected_components(graph):
            if not group&varying:continue
            q=sum(fragment.GetAtomWithIdx(i).GetFormalCharge() for i in group)
            if any(sum(v.GetAtomWithIdx(i).GetFormalCharge() for i in group)!=q for v in variants):raise ChemistryError('Resonance group is not closed under charge transfer')
            elements=[fragment.GetAtomWithIdx(i).GetSymbol() for i in group];nc=elements.count('C');nn=elements.count('N');no=elements.count('O')
            mapped=tuple(sorted(ids[i] for i in group));family=None
            if nc==1 and nn>=2 and no==0:family=('guanidinium' if nn>=3 else 'amidinium') if q>0 else 'amidinate' if q<0 else 'amidine'
            elif nc==1 and nn==1 and no==1:family='amidate' if q<0 else 'amide_resonance'
            elif nc>=2 and nn==0 and no==1:
                family='phenoxide' if any(len(r)>=6 and set(r)<=group for r in fragment.GetRingInfo().AtomRings()) else 'enolate'
            elif nn==1 and no==2 and nc>=1 and any(a.GetSymbol()=='N' and sum(n.GetSymbol()=='O' for n in a.GetNeighbors())==2 for a in fragment.GetAtoms() if a.GetIdx() in group):family='nitronate' if q<0 else 'nitro_resonance'
            for i in group:
                original=ids[i]
                if original not in out or out[original].element not in ('N','O','S','F','Cl','Br','I'):continue
                c=out[original]
                same_local_group=set(c.group_atoms)==set(mapped)
                new_family=c.family if same_local_group else family or 'resonance_'+'_'.join(sorted(set(elements)))
                acceptor=c.acceptor and q<=0
                if q<0 and c.element=='N' and any(v.GetAtomWithIdx(i).GetFormalCharge()<0 for v in variants):acceptor=True
                out[original]=replace(c,family=new_family,group_charge=q,group_atoms=mapped,acceptor=acceptor)
    return out


def _remap(contexts,mapping):
    if any(j not in mapping for c in contexts.values() for j in c.group_atoms):raise ChemistryError('Incomplete chemical group mapping')
    return {mapping[i]:replace(c,group_atoms=tuple(sorted(mapping[j] for j in c.group_atoms))) for i,c in contexts.items()}


def from_generated(mol,rdmols):
    """Use the exact RDKit atom order returned by _build_shell; no inference."""
    ligs=sorted(mol.ligands,key=lambda x:x.ligand_id)
    if len(ligs)!=len(rdmols):raise ChemistryError('RDKit/ligand count mismatch')
    byidx={a.idx:a for a in mol.atoms};out={}
    for lig,rd in zip(ligs,rdmols):
        if rd.GetNumAtoms()!=len(lig.atom_indices):raise ChemistryError('RDKit atom count mismatch')
        mapping={a.GetIdx():idx for a,idx in zip(rd.GetAtoms(),lig.atom_indices) if a.GetSymbol()!='H'}
        if any(a.GetSymbol()!=byidx[idx].symbol for a,idx in zip(rd.GetAtoms(),lig.atom_indices)):raise ChemistryError('RDKit atom order mismatch')
        out.update(_remap(describe(rd),mapping))
    return out


def from_training(mol):
    """Strict element/H-count/bond-order VF2 mapping; no geometric tie-breaks.

    Distinct formal-charge labels on topologically identical fragments remain
    ambiguous in raw topology and are explicitly rejected by the orbit check.
    """
    from jacobik.topology.mapping import _smiles_to_rdkit
    from jacobik.core.chemistry import METALS
    from jacobik.topology.graph import get_ligand_neighbors
    if not mol.smiles:raise ChemistryError('Missing authoritative SMILES')
    _,rd=_smiles_to_rdkit(mol.smiles)
    metals=[a.GetIdx() for a in rd.GetAtoms() if a.GetSymbol() in METALS]
    if len(metals)!=1:raise ChemistryError('Expected one metal in authoritative SMILES')
    rw=Chem.RWMol(rd);rw.RemoveAtom(metals[0]);free=rw.GetMol()
    try:Chem.SanitizeMol(free)
    except Exception as exc:raise ChemistryError('Metal-free SMILES does not sanitize') from exc
    contexts=describe(free);nb=get_ligand_neighbors(mol.atoms,mol.bonds,mol.metal_idx);atom={a.idx:a for a in mol.atoms}
    g=nx.Graph();h=nx.Graph()
    for a in free.GetAtoms():
        if a.GetSymbol()!='H':g.add_node(a.GetIdx(),symbol=a.GetSymbol(),nh=a.GetTotalNumHs(includeNeighbors=True))
    for b in free.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx()
        if i in g and j in g:g.add_edge(i,j,order=round(b.GetBondTypeAsDouble(),3))
    for a in mol.atoms:
        if a.idx!=mol.metal_idx and a.symbol!='H':h.add_node(a.idx,symbol=a.symbol,nh=sum(atom[j].symbol=='H' for j in nb.get(a.idx,())))
    for b in mol.bonds:
        if b.i in h and b.j in h and b.order>=.5:h.add_edge(b.i,b.j,order=round(b.order,3))
    # WL color classes overapproximate automorphism orbits. Requiring invariant
    # contexts within them may reject safe cases, but cannot silently mix charge.
    colors={i:(g.nodes[i]['symbol'],g.nodes[i]['nh']) for i in g}
    for _ in range(len(g)):
        signatures={i:(colors[i],tuple(sorted((g.edges[i,j]['order'],colors[j]) for j in g[i]))) for i in g}
        labels={s:k for k,s in enumerate(sorted(set(signatures.values()),key=repr))};new={i:labels[s] for i,s in signatures.items()}
        if len(set(colors.values()))==len(set(new.values())):colors=new;break
        colors=new
    seen={}
    for i,c in contexts.items():
        # Formal-charge drawings on inactive carbon resonance sites do not
        # affect HB roles and must not veto an otherwise unambiguous mapping.
        token=(c.element,c.family,c.group_charge,c.acceptor,c.donor) if c.acceptor or c.donor else ('inactive',)
        if colors[i] in seen and seen[colors[i]]!=token:raise ChemistryError('Charge/role ambiguous under topology automorphism')
        seen[colors[i]]=token
    mapping=_strict_mapping(g,h)
    return _remap(contexts,mapping)


def _strict_mapping(g,h):
    """Component hashes prevent factorial cross-ligand VF2 backtracking.

    Hashes only reject mismatches; exact VF2 still establishes every mapping.
    The visit budget is computational, not a chemistry or fitted parameter.
    """
    def components(graph):
        out=defaultdict(list)
        for ids in nx.connected_components(graph):
            c=graph.subgraph(ids).copy()
            nx.set_node_attributes(c,{i:f'{c.nodes[i]["symbol"]}:{c.nodes[i]["nh"]}' for i in c},'chem')
            key=nx.weisfeiler_lehman_graph_hash(c,node_attr='chem',edge_attr='order',iterations=3)
            out[key].append(c)
        return out
    gg,hh=components(g),components(h)
    if {k:len(v) for k,v in gg.items()}!={k:len(v) for k,v in hh.items()}:raise ChemistryError('Strict component element/H-count/bond-order mapping failed')
    class BoundedMatcher(nx.algorithms.isomorphism.GraphMatcher):
        visits=0
        def semantic_feasibility(self,i,j):
            self.visits+=1
            if self.visits>50000:raise ChemistryError('Exact component mapping exceeded computational visit budget')
            return super().semantic_feasibility(i,j)
    out={}
    for key,comps in gg.items():
        available=list(hh[key])
        for c in comps:
            for k,d in enumerate(available):
                matcher=BoundedMatcher(c,d,node_match=lambda a,b:(a['symbol'],a['nh'])==(b['symbol'],b['nh']),edge_match=lambda a,b:a['order']==b['order'])
                mapping=next(matcher.isomorphisms_iter(),None)
                if mapping is not None:out.update(mapping);available.pop(k);break
            else:raise ChemistryError('Strict component element/H-count/bond-order mapping failed')
    return out


def ladder(d:AtomContext,a:AtomContext,dbound=False,abound=False):
    """All fallback levels retain both charge signs and acceptor element."""
    dt=f'{d.element}:{d.family}:{d.charge}';at=f'{a.element}:{a.family}:{a.charge}:{a.group_charge}'
    return [('hb',VERSION,dt,at,f'D{int(dbound)}A{int(abound)}'),
            ('hb',VERSION,dt,at,'*'),
            ('hb',VERSION,f'{d.element}:*:{d.charge}',f'{a.element}:*:{a.charge}','*'),
            ('hb',VERSION,f'*:{d.charge}',f'{a.element}:*:{a.charge}','*')]


def candidates(mol,chem):
    from research.archive.code.interactions.nci import hbond_candidates
    from jacobik.topology.graph import donor_indices_of
    bound=set(donor_indices_of(mol));out=[]
    for d,h,a,_ in hbond_candidates(mol):
        if d not in chem or a not in chem:raise ChemistryError('Missing candidate atom chemistry')
        if chem[d].donor and chem[a].acceptor:out.append((d,h,a,ladder(chem[d],chem[a],d in bound,a in bound)))
    return out

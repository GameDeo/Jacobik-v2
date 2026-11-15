"""C-2b seen-seed placement and topology traces; never runs confirmation."""
import argparse
import json
from pathlib import Path
from .phase0 import prepare_parent, seeded_rows, quantiles
from .rings import population


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--only',required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    codes=a.only.split(',');assert set(codes)<=set(population()[0])
    from tools.jgen.bench.run import _set_finish_flags
    from jacobik.ik.fk_compose import ComplexFK
    import torch
    variants=dict(C={},K={'start_kabsch':True},P={'start_kabsch':True,'start_kabsch_pivot':True},
        F={'start_kabsch':True,'start_kabsch_pivot':True,'start_kabsch_diversity':0.})
    for code in codes:
        result=dict(code=code,variants={})
        try:
            _,builder,ctx,assignments,W,T=prepare_parent(code,'cpu')
            from jacobik.topology.decompose import _build_ligand_graph
            import networkx as nx
            result['ligands']=[]
            for li,lig in enumerate(ctx.mol.ligands):
                G,orders=_build_ligand_graph(ctx.mol,lig)
                result['ligands'].append(dict(ligand=li,donors=lig.donor_atoms,
                    macrocycles=lig.macrocycles,rings=[dict(atoms=sorted(r),orders=[orders[tuple(sorted(e))] for e in G.subgraph(r).edges()],
                        fused=G.subgraph(r).number_of_edges()!=len(r)) for r in nx.minimum_cycle_basis(G) if len(r)>=8]))
            for name,flags in variants.items():
                _set_finish_flags(flags)
                data=[]
                for seed in [11,12,13]:
                    rows,caps=seeded_rows(builder,ctx,assignments,W,seed)
                    fks,theta,dpi,ge,_,X,_,_=rows
                    coords=ComplexFK(fks).apply(theta,X)
                    distances=[(ge.donor_positions(coords)[:,di]-kw['donor_targets']).norm(dim=-1)
                               for di,(kw,_) in zip(dpi,caps)]
                    data.append(dict(seed=seed,row_max=quantiles(torch.cat(distances,1).max(1).values.numpy()),
                        per_ligand=[quantiles(d.max(1).values.numpy()) for d in distances],
                        macro_joints=[fk.n_macro_joints for fk in fks]))
                result['variants'][name]=dict(flags=flags,data=data)
            result['status']='ok'
        except Exception as exc:
            result.update(status='failed',reason=repr(exc))
        (a.out/f'{code}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(code,result['status'],flush=True)


if __name__=='__main__':main()

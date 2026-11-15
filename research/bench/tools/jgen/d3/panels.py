"""Registered eligible panels. Never access the auditor's partition file."""
import json
from pathlib import Path
from tools.jgen.c2.rings import MAIN, population


def stage1():return population()[0]


def stage2():
    root=MAIN/'research/data/jgen/campaign/eta2/holdout'
    codes=[c for p in ('ETA2_HOLDOUT.json','ETA2_HOLDOUT_D.json')
           for cs in json.loads((root/p).read_text())['draw'].values() for c in cs]
    if len(codes)!=110 or len(set(codes))!=110:raise ValueError('eta2 frozen holdout must have 110 parents')
    return sorted(codes)


def locked_path(path):
    p=Path(path).resolve()
    if any(a=='.claude' and b=='locked' for a,b in zip(p.parts,p.parts[1:])):
        raise ValueError('access under .claude/locked is forbidden')
    return p


def stage3(codes_file, codes_sha256, analysis_file, run_dir):
    from .arms import sha
    cf=locked_path(codes_file);af=locked_path(analysis_file);rd=locked_path(run_dir)
    if sha(cf)!=codes_sha256:raise ValueError('codes SHA256 mismatch')
    r=json.loads(af.read_text());candidate=r.get('candidate')
    if (r.get('stage')!='stage1' or r.get('label')!='D3' or r.get('reproduction') or candidate not in ('D','DS','S')
        or not (r.get('p_fallback',{}).get('qualified') if candidate=='S' else
                r.get('comparisons',{}).get(candidate,{}).get('qualified'))):
        raise ValueError('stage 1 has no qualified candidate')
    if not (rd/'OWNER_GO').is_file():raise ValueError('missing OWNER_GO')
    codes=cf.read_text().split()
    if len(codes)!=80 or len(set(codes))!=80:raise ValueError('promotion reserve requires 80 distinct supplied codes')
    return codes,candidate

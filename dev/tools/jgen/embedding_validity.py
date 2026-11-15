"""Research-only validity ledger for real GFN2 calls, including cached failures.

Leaves the generator's returned geometry and cache unchanged. The benchmark
rejects a parent after preparation when the source embedding was a fallback.
"""
import hashlib
import logging
import threading
import numpy as np
from rdkit import Chem


class EmbeddingValidity(logging.Handler):
    def __init__(self,engine):
        super().__init__(logging.DEBUG)
        self.engine=engine;self.original=engine._gfn2_refine
        self.parent=None;self.bad_keys={};self.invalid={};self.local=threading.local()

    def emit(self,record):
        message=record.getMessage()
        if 'GFN2 embed refinement failed' in message:
            messages=getattr(self.local,'messages',None)
            if messages is not None:messages.append(message)

    def install(self):
        logging.getLogger().addHandler(self)
        self.engine._gfn2_refine=self.refine
        return self

    def refine(self,mol,positions,donors=None,*,experiment_flags=True):
        data=np.round(np.asarray(positions,dtype=np.float64),4).tobytes()
        key=(Chem.MolToSmiles(mol),len(positions),tuple(sorted({int(a) for g in (donors or []) for a in g})),
             bool(experiment_flags and self.engine.EMBED_BOND_CHECK),bool(experiment_flags and self.engine.EMBED_LI_CAP),hashlib.sha256(data).hexdigest())
        previous=getattr(self.local,'messages',None);self.local.messages=[]
        try:
            value=self.original(mol,positions,donors,experiment_flags=experiment_flags)
            messages=self.local.messages
            if messages:self.bad_keys[key]=list(messages)
            if key in self.bad_keys and self.parent is not None:
                self.invalid.setdefault(self.parent,[]).append({'smiles':key[0],'donors':list(key[2]),'geometry_sha256':key[-1],
                    'cached_failure':not bool(messages),'messages':self.bad_keys[key]})
            return value
        finally:self.local.messages=previous

    def check(self,parent):
        if parent in self.invalid:
            raise RuntimeError(f'GFN2 embedding fallback for {parent}; all generated structures invalid (including cached fallback)')

    def close(self):
        self.engine._gfn2_refine=self.original
        logging.getLogger().removeHandler(self)

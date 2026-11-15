"""Fail-closed pool mapping and explicit provenance-bound overlay interchange.

Geometry loading never requests energy fields. No automatic repair discovery.
"""
import hashlib
import numpy as np

class PoolMappingError(ValueError):
    pass

def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def checked_mapping(symbols, ref_symbols, mapping):
    symbols=np.asarray(symbols);ref_symbols=np.asarray(ref_symbols);g=np.asarray(mapping);n=len(ref_symbols)
    if g.shape!=(n,) or g.dtype.kind not in 'iu' or len(symbols)!=n or not np.array_equal(np.sort(g),np.arange(n)):
        raise PoolMappingError('invalid_atom_map_permutation')
    if not np.array_equal(symbols[g],ref_symbols):raise PoolMappingError('atom_map_symbol_order_mismatch')
    assert (symbols[g]==ref_symbols).all()
    return g.astype(np.int64,copy=False)

def mapped_geometry(pool, *, pool_path=None, overlay_path=None):
    """X[:,g] is REFERENCE ordered. Never substitute identity for a missing map.

    Development overlay NPZ requires source_pool_sha256 and gen_to_ref_idx.
    The actual S2 interchange must be audited before its adapter is enabled.
    """
    repaired='gen_to_ref_idx' not in pool
    if repaired:
        if overlay_path is None:raise PoolMappingError('missing_graph_constrained_atom_map_no_overlay')
        if pool_path is None:raise PoolMappingError('overlay_requires_source_pool_path')
        with np.load(overlay_path,allow_pickle=False) as z:
            if 'source_pool_sha256' not in z or str(z['source_pool_sha256'])!=file_sha256(pool_path):raise PoolMappingError('overlay_source_hash_mismatch')
            g=checked_mapping(pool['symbols'],pool['ref_symbols'],z['gen_to_ref_idx'])
    else:g=checked_mapping(pool['symbols'],pool['ref_symbols'],pool['gen_to_ref_idx'])
    X=np.asarray(pool['X'])
    if X.ndim!=3 or X.shape[1:]!=(len(g),3):raise PoolMappingError('coordinate_shape_mismatch')
    return X[:,g,:],dict(mapping_source='overlay' if repaired else 'pool',overlay_path=str(overlay_path) if repaired else None,original_unmapped=repaired)

def verified_pool_labels(pool, keys, *, pool_path=None, overlay_path=None):
    """Original unmapped MACE/scorer/Q labels are forbidden, even after map repair."""
    _,info=mapped_geometry(pool,pool_path=pool_path,overlay_path=overlay_path)
    if info['original_unmapped'] or 'atom_mapping_error' in pool:
        if overlay_path is None:raise PoolMappingError('unmapped_pool_labels_require_repaired_overlay')
        with np.load(overlay_path,allow_pickle=False) as z:
            if pool_path is None or str(z['source_pool_sha256'])!=file_sha256(pool_path):raise PoolMappingError('overlay_source_hash_mismatch')
            checked_mapping(pool['symbols'],pool['ref_symbols'],z['gen_to_ref_idx'])
            if 'repaired_labels' not in z or not bool(z['repaired_labels']):raise PoolMappingError('overlay_has_no_repaired_labels')
            return {k:np.asarray(z[k]) for k in keys}
    return {k:np.asarray(pool[k]) for k in keys}


def certified_assignment_mask(path, pool_path, nposes):
    """A candidate number, including zero, is never a reference certificate."""
    with np.load(path,allow_pickle=False) as z:
        if str(z['source_pool_sha256'])!=file_sha256(pool_path):raise ValueError('assignment_certificate_pool_hash_mismatch')
        if str(z['schema'])!='nci_v2_assignment_certificate.1':raise ValueError('unknown_assignment_certificate_version')
        mask=np.asarray(z['assignment_matched'])
        if mask.dtype.kind!='b' or mask.shape!=(nposes,):raise ValueError('invalid_assignment_certificate_mask')
        if str(z['certificate_status']) not in ('certified_native','certified_geometric_fallback','uncertifiable'):raise ValueError('unknown_assignment_certificate_status')
        if str(z['certificate_status'])=='uncertifiable' and mask.any():raise ValueError('uncertifiable_assignment_cannot_match')
        return mask

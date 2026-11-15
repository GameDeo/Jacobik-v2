import numpy as np
from tools.jgen.diagnose_nuwfam_cf import geometry


def test_cf_length_and_local_and_ring_plane_tilt():
    t=np.arange(6)*np.pi/3
    ring=np.column_stack([np.cos(t),np.sin(t),np.zeros(6)])
    # A 1.35-A bond elevated 30 degrees above an otherwise planar hexagon.
    fluorine=ring[0]+1.35*np.array([np.cos(np.pi/6),0,np.sin(np.pi/6)])
    g=geometry(np.vstack([ring,fluorine]),[list(range(6))],[(0,6)])
    b=g['bonds'][0]
    assert abs(b['length_A']-1.35)<1e-12
    assert abs(b['ring_plane_tilt_deg']-30)<1e-12
    assert abs(b['local_plane_tilt_deg']-30)<1e-12
    assert abs(b['normal_bond_component_A']-.675)<1e-12
    assert abs(b['F_height_from_ring_plane_A']-.675)<1e-12
    assert g['rings'][0]['rms_height_A']==0

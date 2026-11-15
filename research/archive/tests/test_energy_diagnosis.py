"""Numerical invariants for force/mode diagnosis; no MACE checkpoint required."""
import numpy as np

from tools.jdev.budget.energy_diagnosis import (
    EV, energy_accounting, moved, orthogonal_basis, project_forces,
)


def test_polar_local_energy_keeps_nonlocal_residual():
    local,account=energy_accounting(dict(energy=10.,energies=np.array([2.,3.]),electrostatic_energy=1.))
    np.testing.assert_allclose(local,[2*EV,3*EV])
    assert np.isclose(account['nonlocal_residual'],5*EV)
    assert np.isclose(account['total'],local.sum()+account['nonlocal_residual'])


def test_rotation_preserves_bonds_and_has_correct_tangent():
    X=np.array([[0.,0.,0.],[1.,0.,0.],[0.,2.,0.]])
    origin=np.array([0.,0.,0.]);axis=np.array([0.,0.,1.])
    raw=np.cross(axis,X-origin);scale=np.linalg.norm(raw)
    mode=dict(rows=[0,1,2],axis=axis,origin=origin,rotation_scale=scale,v=raw/scale)
    y=moved(X,mode,.2)
    np.testing.assert_allclose(np.linalg.norm(y[:,None]-y[None],axis=-1),np.linalg.norm(X[:,None]-X[None],axis=-1),atol=1e-12)
    tangent=(moved(X,mode,1e-5)-moved(X,mode,-1e-5))/2e-5
    np.testing.assert_allclose(tangent,mode['v'],atol=1e-10)


def test_force_partitions_do_not_double_count_overlapping_modes():
    f=np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
    a=np.zeros_like(f);a[1,0]=1
    b=np.zeros_like(f);b[2,0]=1
    modes=[dict(kind='translation',v=a),dict(kind='rotation',v=a*2),dict(kind='torsion',v=a+b)]
    result=project_forces(f,modes,0)
    assert np.isclose(sum(result['force_squared'].values()),np.sum(f*f))
    assert np.isclose(result['force_squared']['ligand_rigid'],16.)
    assert np.isclose(result['force_squared']['ligand_torsion'],49.)
    assert np.isclose(sum(result['fraction'].values()),1.)


def test_harmonic_force_projection_sign_and_curvature():
    x=np.array([[1.,-2.,.5]])
    v=np.array([[0.,1.,0.]])
    mode=dict(v=v)
    h=.02
    energy=lambda p: 3*np.sum(p*p)
    force=-6*x
    g=(energy(moved(x,mode,h))-energy(moved(x,mode,-h)))/(2*h)
    curvature=(energy(moved(x,mode,h))+energy(moved(x,mode,-h))-2*energy(x))/h**2
    assert np.isclose(g,-np.sum(force*v))
    assert np.isclose(curvature,6.)


def test_empty_mode_basis_is_valid():
    assert orthogonal_basis([],12).shape==(12,0)


def test_calibrated_contact_display_matches_production_sum_and_gradient():
    import torch
    from jacobik.stats import nci
    from tools.jdev.budget.energy_diagnosis import contact_scores
    r0,dr,nr,a0,da,na=nci._grid('hb')
    f=np.ones((1,nr,na))
    params=nci.build_c2_surface_params('hb',f,s_clip=nci.S_CLIP)
    entry=dict(C=torch.tensor(params['C'],dtype=torch.float64),
               M=torch.tensor(params['M'],dtype=torch.float64),
               T=torch.zeros((1,nr,na),dtype=torch.float64),
               idx=dict(d=torch.tensor([0,0]),h=torch.tensor([1,1]),a=torch.tensor([2,3])),
               row=torch.tensor([0,0]),grid=(r0,dr,nr,a0,da,na),win=nci.SPEC['hb']['win'],
               s_clip=nci.S_CLIP,scale=1.135)
    x=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[3.,0.,0.],[3.2,.2,0.]]],
                   dtype=torch.float64,requires_grad=True)
    displayed=contact_scores('hb',entry,x,'calibrated_hb').sum(-1)
    actual=nci.evaluate({'hb':entry},x)
    torch.testing.assert_close(displayed,actual)
    assert displayed.item()>0
    # Deliberately different T catches accidental reuse of the historical bilinear map.
    assert contact_scores('hb',entry,x,'legacy').sum().item()==0
    g_display,=torch.autograd.grad(displayed.sum(),x,retain_graph=True)
    g_actual,=torch.autograd.grad(actual.sum(),x)
    torch.testing.assert_close(g_display,g_actual)

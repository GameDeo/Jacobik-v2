"""Force maps must rotate with geometry and preserve projected work."""
import numpy as np

from tools.jdev.budget.energy_viewer import align_transform,rotate_mode


def test_alignment_rotates_forces_and_preserves_work():
    x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,2.,0.],[0.,0.,3.]])
    rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    target=x@rot+np.array([3.,-4.,1.])
    R,t=align_transform(x,target,np.arange(len(x)))
    np.testing.assert_allclose(x@R+t,target,atol=1e-12)
    assert np.isclose(np.linalg.det(R),1.)
    force=np.arange(12.).reshape(4,3)
    v=np.flip(force,axis=0)/10
    np.testing.assert_allclose(np.sum((force@R)*(v@R)),np.sum(force*v))


def test_rotational_motion_origin_is_translated_but_axis_is_not():
    R=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    t=np.array([4.,5.,6.])
    m=dict(v=np.array([[1.,2.,3.]]),axis=np.array([0.,0.,1.]),origin=np.array([1.,0.,0.]),rotation_scale=2.)
    out=rotate_mode(m,R,t)
    np.testing.assert_allclose(out['origin'],m['origin']@R+t)
    np.testing.assert_allclose(out['axis'],m['axis']@R)
    np.testing.assert_allclose(out['v'],m['v']@R)
    assert out['rotation_scale']==m['rotation_scale']


def test_alignment_never_reflects_chirality():
    x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,2.,0.],[0.,0.,3.]])
    target=x*np.array([-1.,1.,1.])
    R,t=align_transform(x,target,np.arange(4))
    assert np.linalg.det(R)>.999999
    assert np.linalg.norm(x@R+t-target)>.1

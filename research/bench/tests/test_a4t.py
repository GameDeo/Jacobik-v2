import numpy as np

from tools.jgen.a4t.metrics import (CLASSES, _shell_setup, decision, shell_rmsd,
                                   stereo_diff, stereo_diff_causes, ring_share)


def test_shell_rmsd_reference_and_donor_permutation():
    symbols = ["Fe", "O", "O", "N"]
    X = np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0]],float)
    assert shell_rmsd(X,X,symbols,0,[1,2,3]) < 1e-12
    Y = X.copy(); Y[[1,2]]=Y[[2,1]]
    assert shell_rmsd(Y,X,symbols,0,[1,2,3]) < 1e-12


def test_high_symmetry_shell_caps_without_materializing_factorial():
    symbols=("Fe",)+("C",)*12
    _,perms,capped=_shell_setup(symbols,0,tuple(range(1,13)))
    assert capped and perms.shape==(40320,13)


def test_ordered_decision_list_has_each_class_once():
    args=dict(top_index=0,donor_mismatch=False,geometry_mismatch=False,any_shell=True,
              stereo_mismatch=False,ring_share=0.1)
    rows=[]
    assert decision([dict(hit=True,offset=100)],**args)=="F"
    assert decision([],donor_mismatch=True,**{k:v for k,v in args.items() if k!="donor_mismatch"})=="T"
    assert decision([],geometry_mismatch=True,**{k:v for k,v in args.items() if k!="geometry_mismatch"})=="G"
    assert decision([],geometry_mismatch=False,geometry_evaluable=False,**{k:v for k,v in args.items() if k!="geometry_mismatch"})=="G*"
    assert decision([],any_shell=False,**{k:v for k,v in args.items() if k!="any_shell"})=="S"
    assert decision([],stereo_mismatch=True,**{k:v for k,v in args.items() if k!="stereo_mismatch"})=="Z"
    assert decision([],ring_share=.6,**{k:v for k,v in args.items() if k!="ring_share"})=="R"
    assert decision([],**args)=="C"
    assert set(CLASSES)=={"F","T","G","G*","S","Z","R","C"}


def test_stereo_amendment_ignores_metal_maps_atoms_and_accepts_global_mirror():
    ref=({0:"R",1:"R",2:"S"},{(1,2):"STEREOE"},set())
    row=({0:"S",7:"S",8:"R"},{(7,8):"STEREOE"},set())
    mapping={0:0,7:1,8:2}
    assert stereo_diff(ref,row,mapping,metal=0,allow_mirror=False)
    assert not stereo_diff(ref,row,mapping,metal=0,allow_mirror=True)
    row_unlabelled=({0:"S"},{(7,8):"STEREOE"},set())
    assert not stereo_diff(ref,row_unlabelled,mapping,metal=0,allow_mirror=True)
    assert stereo_diff(ref,row_unlabelled,mapping,metal=0,allow_mirror=True,
                       require_reference_centres=True)
    row_partial_mirror=({0:"S",7:"S",8:"S"},{(7,8):"STEREOE"},set())
    assert stereo_diff(ref,row_partial_mirror,mapping,metal=0,allow_mirror=True)


def test_ring_share_returns_its_rmsd_optimal_atom_mapping():
    symbols=["Fe","O","O","N","F"]
    X=np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,0,1]],float)
    P=np.array([[0,1,2,3,4],[0,2,1,3,4]])
    value,mapping=ring_share(X[[0,2,1,3,4]],X,symbols,P,set(),return_atom_map=True)
    assert value==0.0
    assert mapping[1]==2 and mapping[2]==1


def test_stereoany_or_absent_double_bond_label_is_not_compared():
    ref=({}, {(1,2):"STEREOE", (3,4):"STEREOANY", (5,6):"STEREOZ"}, set())
    row=({}, {(1,2):"STEREOANY", (3,4):"STEREOE"}, set())
    assert not stereo_diff(ref,row,allow_mirror=False)
    assert stereo_diff(ref,row,allow_mirror=False,ignore_undetermined=False)


def test_cause_is_donor_only_when_a_differing_centre_is_a_donor():
    ref=({0:"R",1:"R",2:"R"},{},set())
    row=({0:"S",1:"R",2:"S"},{},set())
    _,_,remaining=stereo_diff_causes(ref,row,{0:0,1:1,2:2},0,{1})
    assert remaining==["ligand centre"]

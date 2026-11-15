import pytest
import numpy as np
from tools.jgen.a4b.relax_set import select
from tools.jgen.a4b.tables import unrel_bucket, rel_bucket, relaxed_values, unrel_metrics
from tools.jgen.a4b.relax_run import to_ref_order

def test_select_top_floor_and_nearest_distinct_rows():
    m={'n_rows':6,'rows':[{'is_top1':i==4,'is_floor':i==5} for i in range(6)]}
    r={'n_rows':6,'coverage':{'rmsd_A':[.2,.1,.3,1.6,2.,2.]}}
    assert select(m,r)==[0,1,2,4,5]

def test_select_rejects_bad_row_count():
    with pytest.raises(ValueError): select({'n_rows':1,'rows':[]},{'n_rows':2})

def test_frozen_bucket_rules():
    assert unrel_bucket([1.0, 1.2, 2.0], 1)=='U-POOL'
    assert unrel_bucket([1.0, 1.2], 0)=='U-TOP1'
    assert unrel_bucket([1.1, 1.2], 0)=='U-NONE'
    assert rel_bucket({0:{'hit':False,'equivalent':False},1:{'hit':True,'equivalent':False}},0)=='R-RANK'
    assert rel_bucket({0:{'hit':True,'equivalent':False}},0)=='R-TOP1'

def test_relaxed_identity():
    x=relaxed_values(20.,12.,0.,3.)
    assert x['strain']==8 and x['offset']==9 and x['offset_raw']==12

def test_unrelaxed_pair_inversion_and_obscene_flags():
    m=unrel_metrics([0.3,0.9,0.8],[0.,150.,10.],[2.,2.,2.],0,0,0.)
    assert m['ranking_loss']==0.
    assert m['obscene'] is False
    assert m['top1_inversion'] is False
    assert m['pair_inversion_rate']==1.
    m=unrel_metrics([0.1,0.9],[250.,0.],[2.,2.],0,1,0.)
    assert m['obscene'] is True and m['top1_inversion'] is True

def test_builder_to_reference_atom_order_round_trip():
    ref=[[10,0,0],[20,0,0],[30,0,0]]
    # Manifest mapping lists builder indices in reference order.
    mapping=[2,0,1]
    gen=[ref[1],ref[2],ref[0]]
    assert to_ref_order(gen,mapping).tolist()==ref
    assert np.array(ref)[np.argsort(mapping)].tolist()==gen

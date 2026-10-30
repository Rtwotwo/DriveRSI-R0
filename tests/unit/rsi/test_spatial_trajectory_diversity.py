import numpy as np
import pytest
from driversi.rsi.decisions.trajectory_diversity import compare, path_distances
from driversi.rendering.diagnostics.numeric_admission import numeric_settings,strict_problem_profile


def descriptor(xy,name):
    return dict(comparison_key={'source':'same'},candidate_id=name,xy=xy.tolist())


def test_speed_change_alone_fails_spatial_novelty():
    a=np.column_stack([np.linspace(0,20,71),np.zeros(71)])
    b=np.column_stack([np.linspace(0,30,71),np.zeros(71)])
    aa,bb=descriptor(a,'a'),descriptor(b,'b')
    assert compare(bb,[aa])['distinct']
    decision=compare(bb,[aa],minimum_path_ade_m=.5)
    assert not decision['distinct']
    assert decision['comparisons'][0]['path_ade_m']==pytest.approx(0)


def test_actual_lateral_path_change_passes_both_tests():
    x=np.linspace(0,20,71)
    a=np.column_stack([x,np.zeros(71)])
    b=np.column_stack([x,np.minimum(x,1.)])
    assert compare(descriptor(b,'b'),[descriptor(a,'a')],minimum_path_ade_m=.5)['distinct']


def test_stops_do_not_create_path_novelty():
    a=np.column_stack([np.linspace(0,20,71),np.zeros(71)])
    b=a.copy();b[:21,0]=0;b[21:,0]=np.linspace(0,20,50)
    assert path_distances(a,b)['path_ade_m']==pytest.approx(0)


def test_insufficient_common_path_is_not_spatial_evidence():
    a=np.zeros((71,2));b=a.copy();b[:,1]=np.linspace(0,1,71)
    assert path_distances(a,b)['path_ade_m'] is None
    assert not compare(descriptor(b,'b'),[descriptor(a,'a')],minimum_path_ade_m=.5)['distinct']


def test_short_first_reference_cannot_poison_the_family():
    stopped=descriptor(np.zeros((71,2)),'stopped')
    assert not compare(stopped,[],minimum_path_ade_m=.5)['distinct']
    moving=descriptor(np.column_stack([np.linspace(0,20,71),np.zeros(71)]),'moving')
    assert compare(moving,[],minimum_path_ade_m=.5)['distinct']


def test_new_profile_is_separate_from_legacy25_and27():
    profile='experimental_numeric25_spatial05_v1'
    assert numeric_settings(profile)['minimum_path_ade_m']==.5
    assert strict_problem_profile(profile,25)
    assert not strict_problem_profile(profile,27)
    assert not strict_problem_profile('experimental_numeric25_v1',25)
    assert 'minimum_path_ade_m' not in numeric_settings('experimental_numeric27_strict_ade05_v1')


def test_spatial_prefix_preserves_envelope_and_balances_speed():
    from driversi.rsi.experiments.generation_policy import spatial_grid,symmetric_grid
    grid=spatial_grid();key=lambda r:tuple(sorted(r.items()))
    assert set(map(key,grid))==set(map(key,symmetric_grid()))
    assert all(r['lateral_m']!=0 for r in grid[:32])
    for n in range(10,65):assert {r['speed_delta_mps'] for r in grid[:n]}=={-1.,0.,1.,2.}

# SPDX-License-Identifier: Apache-2.0
import importlib
from pathlib import Path
import math
import pytest
import torch
from vapor import VAPORConfig
from vapor.reference import visual_factors, visual_mass, proxy_mass, age_threshold
from vapor.prefill import make_query_chunked_eager

@pytest.mark.parametrize('family,cut,low,high',[('qwen',.75,.20,.10),('internvl',.70,.15,.05)])
def test_threshold_boundaries(family,cut,low,high):
    c=VAPORConfig() if family=='qwen' else VAPORConfig.internvl()
    assert c.select_threshold(cut-1e-8)==low
    assert c.select_threshold(cut)==high
    assert c.select_threshold(1)==high

@pytest.mark.parametrize('kwargs',[{'rank':3},{'cache_age':2},{'max_new_tokens':9000},
                                  {'threshold':float('nan')},{'prefill_query_block':0}])
def test_bad_config(kwargs):
    with pytest.raises(ValueError):VAPORConfig(**kwargs)

@pytest.mark.parametrize('tokens,dim,rank',[(5,8,5),(17,8,8),(2,8,2)])
def test_full_rank_recovers_exact_mass(tokens,dim,rank):
    g=torch.Generator().manual_seed(31)
    k=torch.randn(3,tokens,dim,generator=g);q=torch.randn(3,dim,generator=g)
    text=torch.randn(3,9,generator=g)
    low,basis=visual_factors(k,rank)
    exact=visual_mass((k@q.unsqueeze(-1)).squeeze(-1)/math.sqrt(dim),text)
    torch.testing.assert_close(proxy_mass(q,low,basis,text),exact,atol=2e-6,rtol=2e-6)

@pytest.mark.parametrize('rank',[1,2,4,8])
def test_mass_error_bound(rank):
    torch.manual_seed(9)
    k=torch.randn(3,13,8);q=torch.randn(3,8);text=torch.randn(3,7)
    low,basis=visual_factors(k,rank)
    approx=low@basis.transpose(-1,-2)
    residual=torch.linalg.matrix_norm(k-approx,ord=2)
    epsilon=residual*q.norm(dim=-1)/math.sqrt(8)
    exact=visual_mass((k@q.unsqueeze(-1)).squeeze(-1)/math.sqrt(8),text)
    estimated=proxy_mass(q,low,basis,text)
    assert torch.all((exact-estimated).abs()<=torch.tanh(epsilon/4)+2e-6)


def test_age_decay_boundaries():
    assert age_threshold(.1,1,512)==.1
    assert age_threshold(.1,2,512)==.1
    assert age_threshold(.1,512,512)==pytest.approx(0,abs=1e-15)
    assert age_threshold(.1,128,512)>age_threshold(.1,256,512)


def test_query_chunking_keeps_all_keys_and_mask():
    from types import SimpleNamespace
    torch.manual_seed(3);q=torch.randn(1,2,11,8);k=torch.randn_like(q);v=torch.randn_like(q)
    mask=torch.full((1,1,11,11),float('-inf')).triu(1)
    owner=SimpleNamespace(training=False);key_counts=[]
    def eager(module,q,k,v,mask,scaling,dropout=0.,**kw):
        key_counts.append(k.shape[-2])
        p=(q@k.transpose(-1,-2)*scaling+mask).softmax(-1)
        return (p@v).transpose(1,2).contiguous(),p
    full=eager(owner,q,k,v,mask,.3)[0]
    out,weights=make_query_chunked_eager(eager,3,4)(owner,q,k,v,mask,.3)
    torch.testing.assert_close(full,out)
    assert weights is None and key_counts==[11]*5


def test_import_dependency_closure():
    import vapor
    root=Path(vapor.__file__).parent
    for p in (root/'_backend').rglob('*.py'):
        if p.name!='__init__.py':
            importlib.import_module('vapor.'+'.'.join(p.relative_to(root).with_suffix('').parts))


def test_kernel_launch_helpers_do_not_compile_on_import():
    from vapor._backend.qwen import parallel_proxy_batch_dispatch as q
    from vapor._backend.internvl import method_pipeline_graph_dispatch as i
    assert q._EXTENSION is None and i._EXTENSION is None


def test_native_scalar_gate_threshold_tie_refreshes_now():
    from vapor._backend.qwen.lean_stats_gate import SVDCheckpointHoldGate
    b=torch.tensor([True,True]);ready=torch.ones(2,dtype=torch.bool)
    result=SVDCheckpointHoldGate._transition(b,b,torch.tensor([.099,.1]),torch.tensor([.1,.1]),
                  torch.tensor([float('nan'),.12]),.1,ready,torch.tensor([3,3]),512)
    assert result['stay_hold'].tolist()==[True,False]
    assert result['dense_now'].tolist()==[False,True]

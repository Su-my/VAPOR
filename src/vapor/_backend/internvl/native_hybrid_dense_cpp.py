# SPDX-License-Identifier: Apache-2.0
# Adapted from research source for package-relative imports.
import hashlib
import os
from pathlib import Path
SOURCE = '#include <torch/extension.h>\n#include <optional>\n#include <vector>\nstd::vector<at::Tensor> dense_parts(const at::Tensor& query,const at::Tensor& key,const at::Tensor& value,\n const std::optional<at::Tensor>& mask,const at::Tensor& positions,const at::Tensor& routes,int64_t groups,double scaling){\n auto H=query.size(1),D=query.size(3),N=key.size(2);\n auto heads=at::nonzero(routes.select(0,0)).flatten();\n auto mass=at::full({H},NAN,query.options().dtype(at::kFloat));\n auto visual=at::full({H,D},NAN,query.options().dtype(at::kFloat));\n auto denseout=at::zeros_like(query);\n    if(heads.numel()) {\n        auto ki=at::floor_divide(heads,groups);\n        auto keys=key.select(0,0).index_select(0,ki),values=value.select(0,0).index_select(0,ki);\n        auto q=query.select(0,0).select(1,0).index_select(0,heads).unsqueeze(1);\n        auto logits=at::matmul(q,keys.transpose(1,2))*scaling;\n        if(mask.has_value()&&mask->defined())logits+=mask->select(0,0).select(0,0).slice(0,0,1).slice(1,0,N);\n        auto probs=at::softmax(logits,-1,at::kFloat).to(query.scalar_type());\n        denseout.select(0,0).select(1,0).index_copy_(0,heads,at::matmul(probs,values).squeeze(1));\n        auto vp=probs.select(1,0).index_select(1,positions).to(at::kFloat);\n        auto vv=values.index_select(1,positions).to(at::kFloat);\n        auto vm=vp.sum(-1),vr=at::einsum("hv,hvd->hd",{vp,vv});\n        mass.index_copy_(0,heads,vm);\n        visual.index_copy_(0,heads,at::where(vm.unsqueeze(1)>0,vr/at::clamp_min(vm,1.e-20).unsqueeze(1),at::zeros_like(vr)));\n    }\n return {denseout.reshape({H,D}),mass,visual};\n}\n'
_MODULE = None

def load():
    global _MODULE
    if _MODULE is None:
        from torch.utils.cpp_extension import load_inline
        for p in ['/opt/rh/gcc-toolset-12/root/usr/bin/g++', '/opt/rh/gcc-toolset-11/root/usr/bin/g++']:
            if Path(p).exists():
                os.environ['CXX'] = p
                break
        os.environ['MAX_JOBS'] = '1'
        _MODULE = load_inline(name='phase_native_hybrid_dense_' + hashlib.sha256(SOURCE.encode()).hexdigest()[:16], cpp_sources=SOURCE, functions=['dense_parts'], extra_cflags=['-O3'], with_cuda=False, verbose=False)
    return _MODULE

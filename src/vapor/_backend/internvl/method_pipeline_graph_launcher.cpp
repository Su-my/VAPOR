#include <Python.h>
#include <cuda.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

union Value { uint64_t u; int64_t i; float f; double d; int8_t i8; int16_t i16; uint8_t u8; uint16_t u16; };
struct Slot { char kind; Value value{}; };
struct Update { int slot; int input; };
struct Packet {
    uint64_t function; int warps, shared, grid_x, divisor;
    std::vector<Slot> slots; std::vector<void*> pointers; std::vector<Update> updates, patches;
};
using GraphDestroy=CUresult(*)(CUgraph);
using ExecDestroy=CUresult(*)(CUgraphExec);
static GraphDestroy graph_destroy=nullptr;static ExecDestroy exec_destroy=nullptr;
struct Batch {
    std::vector<Packet> packets;CUgraph graph=nullptr;CUgraphExec executable=nullptr;std::vector<CUgraphNode> nodes;
    ~Batch(){if(executable&&exec_destroy)exec_destroy(executable);if(graph&&graph_destroy)graph_destroy(graph);}
};
static constexpr const char* capsule_name="phase_hold_batch";

static bool assign(Slot& slot, PyObject* obj) {
    switch (slot.kind) {
    case 'p': case 'U': slot.value.u=PyLong_AsUnsignedLongLong(obj); break;
    case 'L': slot.value.i=PyLong_AsLongLong(obj); break;
    case 'i': case 's': case 'b': {
        long long v=PyLong_AsLongLong(obj);
        long long lo=slot.kind=='b'?-128:slot.kind=='s'?-32768:std::numeric_limits<int32_t>::min();
        long long hi=slot.kind=='b'?127:slot.kind=='s'?32767:std::numeric_limits<int32_t>::max();
        if (!PyErr_Occurred() && (v<lo || v>hi)) { PyErr_SetString(PyExc_OverflowError,"Signed kernel argument overflow"); return false; }
        if (slot.kind=='b') slot.value.i8=v; else if(slot.kind=='s') slot.value.i16=v; else slot.value.i=v;
        break;
    }
    case 'u': case 'S': case 'B': {
        uint64_t v=PyLong_AsUnsignedLongLong(obj),hi=slot.kind=='B'?255:slot.kind=='S'?65535:std::numeric_limits<uint32_t>::max();
        if (!PyErr_Occurred() && v>hi) { PyErr_SetString(PyExc_OverflowError,"Unsigned kernel argument overflow"); return false; }
        if(slot.kind=='B')slot.value.u8=v; else if(slot.kind=='S')slot.value.u16=v; else slot.value.u=v;
        break;
    }
    case 'f': slot.value.f=static_cast<float>(PyFloat_AsDouble(obj)); break;
    case 'd': slot.value.d=PyFloat_AsDouble(obj); break;
    default: PyErr_SetString(PyExc_ValueError,"Unsupported kernel argument kind"); return false;
    }
    return !PyErr_Occurred();
}
static void destroy(PyObject* capsule) { delete static_cast<Batch*>(PyCapsule_GetPointer(capsule,capsule_name)); }
static PyObject* create(PyObject*,PyObject* descriptions) {
    PyObject* sequence=PySequence_Fast(descriptions,"Kernel descriptions must be a sequence"); if(!sequence)return nullptr;
    auto* batch=new Batch;
    for(Py_ssize_t i=0;i<PySequence_Fast_GET_SIZE(sequence);++i) {
        PyObject *values,*updates,*patches=Py_None; const char* kinds; unsigned long long function; int warps,shared,grid_x,divisor;
        if(!PyArg_ParseTuple(PySequence_Fast_GET_ITEM(sequence,i),"KiiiisOO|O",&function,&warps,&shared,&grid_x,&divisor,&kinds,&values,&updates,&patches))goto fail;
        if(warps<=0 || shared<0 || grid_x<=0 || divisor==std::numeric_limits<int32_t>::min()){PyErr_SetString(PyExc_ValueError,"Invalid launch geometry");goto fail;}
        PyObject* args=PySequence_Fast(values,"Kernel values must be a sequence");if(!args)goto fail;
        if(PySequence_Fast_GET_SIZE(args)!=static_cast<Py_ssize_t>(std::strlen(kinds))){Py_DECREF(args);PyErr_SetString(PyExc_ValueError,"Kernel type/value lengths differ");goto fail;}
        Packet packet;packet.function=function;packet.warps=warps;packet.shared=shared;packet.grid_x=grid_x;packet.divisor=divisor;
        for(Py_ssize_t j=0;j<PySequence_Fast_GET_SIZE(args);++j){
            Slot slot;slot.kind=kinds[j];if(!assign(slot,PySequence_Fast_GET_ITEM(args,j))){Py_DECREF(args);goto fail;}
            packet.slots.push_back(slot);
        }
        Py_DECREF(args);
        // Triton 3.4 appends one global-scratch pointer even when its size is zero.
        Slot scratch;scratch.kind='p';packet.slots.push_back(scratch);
        PyObject* changes=PySequence_Fast(updates,"Kernel updates must be a sequence");if(!changes)goto fail;
        for(Py_ssize_t j=0;j<PySequence_Fast_GET_SIZE(changes);++j){
            int slot,input;if(!PyArg_ParseTuple(PySequence_Fast_GET_ITEM(changes,j),"ii",&slot,&input)){Py_DECREF(changes);goto fail;}
            if(slot<0 || slot>=static_cast<int>(packet.slots.size())-1 || input<0 || input>=10){Py_DECREF(changes);PyErr_SetString(PyExc_ValueError,"Invalid dynamic argument index");goto fail;}
            packet.updates.push_back({slot,input});
        }
        Py_DECREF(changes);
        if(patches!=Py_None){
            PyObject* changes=PySequence_Fast(patches,"Fixed patch list required");if(!changes)goto fail;
            for(Py_ssize_t j=0;j<PySequence_Fast_GET_SIZE(changes);++j){
                int slot,input;if(!PyArg_ParseTuple(PySequence_Fast_GET_ITEM(changes,j),"ii",&slot,&input)){Py_DECREF(changes);goto fail;}
                if(slot<0 || slot>=static_cast<int>(packet.slots.size())-1 || input<0 || input>=7){Py_DECREF(changes);PyErr_SetString(PyExc_ValueError,"Invalid layer/state patch");goto fail;}
                packet.patches.push_back({slot,input});
            }
            Py_DECREF(changes);
        }
        batch->packets.push_back(std::move(packet));
    }
    Py_DECREF(sequence);
    for(auto& packet:batch->packets)for(auto& slot:packet.slots)packet.pointers.push_back(&slot.value);
    return PyCapsule_New(batch,capsule_name,destroy);
fail: Py_DECREF(sequence);delete batch;return nullptr;
}
static PyObject* clone(PyObject*,PyObject* args){
    PyObject *capsule,*values;if(!PyArg_ParseTuple(args,"OO",&capsule,&values))return nullptr;
    auto* source=static_cast<Batch*>(PyCapsule_GetPointer(capsule,capsule_name));if(!source)return nullptr;
    PyObject* patches=PySequence_Fast(values,"Layer/state patch values required");if(!patches)return nullptr;
    if(PySequence_Fast_GET_SIZE(patches)!=7){Py_DECREF(patches);PyErr_SetString(PyExc_ValueError,"Expected six state pointers and layer offset");return nullptr;}
    auto* batch=new Batch;batch->packets=source->packets;
    for(auto& packet:batch->packets){
        packet.pointers.clear();
        for(auto& change:packet.patches)
            if(!assign(packet.slots[change.slot],PySequence_Fast_GET_ITEM(patches,change.input))){delete batch;Py_DECREF(patches);return nullptr;}
        for(auto& slot:packet.slots)packet.pointers.push_back(&slot.value);
    }
    Py_DECREF(patches);return PyCapsule_New(batch,capsule_name,destroy);
}
static bool update(Batch& batch,PyObject* values) {
    PyObject* args=PySequence_Fast(values,"Dynamic values must be a sequence");if(!args)return false;
    if(PySequence_Fast_GET_SIZE(args)!=10){Py_DECREF(args);PyErr_SetString(PyExc_ValueError,"Expected ten HOLD dynamic values");return false;}
    for(auto& packet:batch.packets)for(auto& change:packet.updates)
        if(!assign(packet.slots[change.slot],PySequence_Fast_GET_ITEM(args,change.input))){Py_DECREF(args);return false;}
    Py_DECREF(args);return true;
}
static PyObject* inspect_slots(PyObject*,PyObject* args) {
    PyObject *capsule,*values;if(!PyArg_ParseTuple(args,"OO",&capsule,&values))return nullptr;
    auto* batch=static_cast<Batch*>(PyCapsule_GetPointer(capsule,capsule_name));if(!batch || !update(*batch,values))return nullptr;
    PyObject* outer=PyList_New(batch->packets.size());
    for(size_t i=0;i<batch->packets.size();++i){
        auto& packet=batch->packets[i];PyObject* row=PyList_New(packet.slots.size());
        for(size_t j=0;j<packet.slots.size();++j){auto& s=packet.slots[j];int n=(s.kind=='f'||s.kind=='i'||s.kind=='u')?4:(s.kind=='s'||s.kind=='S')?2:(s.kind=='b'||s.kind=='B')?1:8;
            PyList_SET_ITEM(row,j,PyBytes_FromStringAndSize(reinterpret_cast<const char*>(&s.value),n));}
        PyList_SET_ITEM(outer,i,row);
    }
    return outer;
}
using Error=CUresult(*)(CUresult,const char**);
using GraphCreate=CUresult(*)(CUgraph*,unsigned int);
using AddKernel=CUresult(*)(CUgraphNode*,CUgraph,const CUgraphNode*,size_t,const CUDA_KERNEL_NODE_PARAMS*);
using Instantiate=CUresult(*)(CUgraphExec*,CUgraph,unsigned long long);
using SetKernel=CUresult(*)(CUgraphExec,CUgraphNode,const CUDA_KERNEL_NODE_PARAMS*);
using LaunchGraph=CUresult(*)(CUgraphExec,CUstream);
static void* cuda_library=nullptr;static Error error_string=nullptr;
static GraphCreate graph_create=nullptr;static AddKernel add_kernel=nullptr;static Instantiate instantiate=nullptr;
static SetKernel set_kernel=nullptr;static LaunchGraph launch_graph=nullptr;
static PyObject* run(PyObject*,PyObject* args) {
    PyObject *capsule,*values;unsigned long long raw_stream;
    if(!PyArg_ParseTuple(args,"OOK",&capsule,&values,&raw_stream))return nullptr;
    auto* batch=static_cast<Batch*>(PyCapsule_GetPointer(capsule,capsule_name));if(!batch || !update(*batch,values))return nullptr;
    if(!cuda_library){
        cuda_library=dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);if(!cuda_library){PyErr_SetString(PyExc_RuntimeError,dlerror());return nullptr;}
        error_string=reinterpret_cast<Error>(dlsym(cuda_library,"cuGetErrorString"));
        graph_create=reinterpret_cast<GraphCreate>(dlsym(cuda_library,"cuGraphCreate"));
        add_kernel=reinterpret_cast<AddKernel>(dlsym(cuda_library,"cuGraphAddKernelNode"));
        instantiate=reinterpret_cast<Instantiate>(dlsym(cuda_library,"cuGraphInstantiateWithFlags"));
        set_kernel=reinterpret_cast<SetKernel>(dlsym(cuda_library,"cuGraphExecKernelNodeSetParams"));
        launch_graph=reinterpret_cast<LaunchGraph>(dlsym(cuda_library,"cuGraphLaunch"));
        graph_destroy=reinterpret_cast<GraphDestroy>(dlsym(cuda_library,"cuGraphDestroy"));
        exec_destroy=reinterpret_cast<ExecDestroy>(dlsym(cuda_library,"cuGraphExecDestroy"));
    }
    if(!graph_create||!add_kernel||!instantiate||!set_kernel||!launch_graph||!graph_destroy||!exec_destroy){PyErr_SetString(PyExc_RuntimeError,"CUDA graph symbols unavailable");return nullptr;}
    if(batch->packets.size()!=8 && batch->packets.size()!=9){PyErr_SetString(PyExc_ValueError,"Only eight or nine method kernels may enter this graph");return nullptr;}
    PyObject* n_obj=PySequence_GetItem(values,6);if(!n_obj)return nullptr;long n=PyLong_AsLong(n_obj);Py_DECREF(n_obj);if(PyErr_Occurred())return nullptr;
    if(n<=0 || n>std::numeric_limits<int32_t>::max()){PyErr_SetString(PyExc_ValueError,"Invalid KV length");return nullptr;}
    CUresult status=CUDA_SUCCESS;size_t failed=0;const char* stage="create";
    Py_BEGIN_ALLOW_THREADS
    bool initializing=batch->graph==nullptr;
    if(initializing)status=graph_create(&batch->graph,0);
    if(status==CUDA_SUCCESS){
        for(size_t i=0;i<batch->packets.size();++i){
            auto& packet=batch->packets[i];CUDA_KERNEL_NODE_PARAMS config{};
            config.func=reinterpret_cast<CUfunction>(packet.function);
            config.gridDimX=packet.grid_x;config.gridDimY=packet.divisor>0?(n+packet.divisor-1)/packet.divisor:packet.divisor<0?-packet.divisor:1;config.gridDimZ=1;
            config.blockDimX=32*packet.warps;config.blockDimY=1;config.blockDimZ=1;config.sharedMemBytes=packet.shared;
            config.kernelParams=packet.pointers.data();config.extra=nullptr;
            if(initializing){
                stage="add";CUgraphNode node=nullptr;
                status=add_kernel(&node,batch->graph,i?&batch->nodes.back():nullptr,i?1:0,&config);
                if(status==CUDA_SUCCESS)batch->nodes.push_back(node);
            }else{stage="update";status=set_kernel(batch->executable,batch->nodes[i],&config);}
            if(status!=CUDA_SUCCESS){failed=i;break;}
        }
    }
    if(status==CUDA_SUCCESS&&initializing){stage="instantiate";status=instantiate(&batch->executable,batch->graph,0);}
    if(status==CUDA_SUCCESS){stage="launch";status=launch_graph(batch->executable,reinterpret_cast<CUstream>(raw_stream));}
    Py_END_ALLOW_THREADS
    if(status!=CUDA_SUCCESS){const char* text="Unknown CUDA error";if(error_string)error_string(status,&text);PyErr_Format(PyExc_RuntimeError,"HOLD graph %s at kernel %zu failed: %s",stage,failed,text);return nullptr;}
    Py_RETURN_NONE;
}
static PyMethodDef methods[]={{"create",create,METH_O,nullptr},{"clone",clone,METH_VARARGS,nullptr},{"run",run,METH_VARARGS,nullptr},{"inspect_slots",inspect_slots,METH_VARARGS,nullptr},{nullptr,nullptr,0,nullptr}};
static PyModuleDef module={PyModuleDef_HEAD_INIT,"_phase_hold_batch",nullptr,-1,methods};
PyMODINIT_FUNC PyInit__phase_hold_batch(){return PyModule_Create(&module);}

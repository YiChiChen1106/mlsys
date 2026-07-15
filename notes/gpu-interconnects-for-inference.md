# GPU Interconnects For LLM Inference

This note explains PCIe, NVLink, NVSwitch, NIC, RDMA, GPUDirect RDMA, and NCCL from an LLM inference systems perspective.

Beginner mental model:

```text
PCIe / NVLink / NVSwitch:
mostly inside one machine, connecting GPUs and CPU/root complex

NIC / InfiniBand / RoCE / RDMA:
mostly between machines, connecting nodes over a network

GPUDirect RDMA:
lets RDMA-capable NICs directly read/write GPU memory without staging through CPU memory

NCCL:
software communication library that uses available hardware paths for GPU collectives and P2P
```

## Why This Matters For Inference

Modern LLM inference is not only matrix multiplication. It also moves data:

- tensor-parallel all-reduce,
- pipeline-parallel activation transfer,
- expert-parallel all-to-all,
- prefill-decode KV cache transfer,
- distributed KV cache read/write,
- model weight loading,
- metrics/logging/control-plane traffic.

When communication is slow, more GPUs do not automatically mean faster serving.

This explains our TP=2 result on `pink`:

```text
2 x RTX 4090 helped throughput
but speedup stayed below 2x
because GPU0-GPU1 topology was SYS, not NVLink
```

## PCIe

PCIe is the general-purpose peripheral interconnect in a server.

GPUs, NICs, and NVMe drives usually connect through PCIe.

In a dual consumer-GPU machine like `pink`, the two RTX 4090s usually communicate through PCIe and the CPU/root complex rather than NVLink.

Beginner mental model:

```text
PCIe = main server highway for devices
```

Why it matters:

- tensor parallel communication can go through PCIe if there is no NVLink,
- PCIe topology can involve CPU/root-complex hops,
- GPU-to-GPU communication can be slower than on NVLink/NVSwitch systems,
- `nvidia-smi topo -m` helps reveal the topology.

Common topology labels:

```text
NV#  = connected by NVLink
PIX  = through one PCIe switch
PXB  = through multiple PCIe switches
PHB  = through PCIe host bridge / CPU
SYS  = across system interconnect, often involving CPU sockets/NUMA
```

For LLM inference, a `SYS` path is a warning that multi-GPU communication may be expensive.

## NVLink

NVLink is NVIDIA's high-bandwidth GPU interconnect.

Beginner mental model:

```text
NVLink = faster GPU-to-GPU road than PCIe
```

It is useful for:

- tensor parallel all-reduce,
- pipeline parallel activation transfer,
- multi-GPU attention or cache movement,
- high-frequency GPU-to-GPU communication.

If two GPUs are connected by NVLink, tensor parallelism is usually more attractive because cross-GPU communication is cheaper.

On `pink`, the two RTX 4090s were reported as `SYS`, so we should not expect NVLink-class scaling.

## NVSwitch

NVSwitch extends NVLink into a switch fabric.

Beginner mental model:

```text
NVLink = fast GPU-to-GPU link
NVSwitch = switch fabric connecting many GPUs with NVLink
```

NVSwitch is common in datacenter systems where many GPUs need high-bandwidth all-to-all communication.

It is especially useful for:

- large tensor parallel groups,
- expert parallel all-to-all,
- multi-GPU inference with frequent collectives,
- training clusters.

## NIC

NIC means network interface card.

In AI servers, this often means:

- InfiniBand adapter,
- RoCE-capable Ethernet adapter,
- NVIDIA ConnectX NIC,
- DPU/SuperNIC in more advanced systems.

Beginner mental model:

```text
NIC = network card that connects this server to other servers
```

For single-node inference, NIC is less central.

For multi-node inference, NIC becomes important for:

- KV cache transfer between prefill and decode nodes,
- distributed serving,
- expert parallel communication,
- model parallelism across nodes,
- disaggregated KV cache storage.

## RDMA

RDMA means Remote Direct Memory Access.

It allows one machine to read or write another machine's memory over the network with low CPU involvement.

Beginner mental model:

```text
normal network path:
remote data -> NIC -> CPU/kernel/network stack -> memory -> GPU

RDMA:
remote data -> NIC -> target memory with much less CPU involvement
```

RDMA is useful for low-latency, high-bandwidth communication across nodes.

In LLM inference, RDMA matters when:

- prefill and decode are on different nodes,
- KV cache is transferred across machines,
- expert-parallel tokens move across nodes,
- distributed cache systems need low-latency data movement.

## GPUDirect RDMA

GPUDirect RDMA lets an RDMA-capable NIC directly access GPU memory.

Beginner mental model:

```text
without GPUDirect RDMA:
GPU memory -> CPU host memory -> NIC -> network

with GPUDirect RDMA:
GPU memory -> NIC -> network
```

The point is to avoid extra copies through CPU memory and reduce CPU overhead.

This is very relevant for PD disaggregation:

```text
prefill GPU builds KV cache
decode GPU needs KV cache
GPUDirect RDMA can help transfer GPU-resident KV cache across nodes
```

But it requires the right hardware and software stack:

- RDMA-capable NIC,
- compatible GPU/platform topology,
- drivers and runtime support,
- often InfiniBand or RoCE,
- careful memory registration and communication design.

It is not automatically available just because a server has GPUs.

## NCCL

NCCL is NVIDIA's communication library for GPUs.

It provides:

- all-reduce,
- all-gather,
- reduce-scatter,
- broadcast,
- send/receive,
- other collective and P2P primitives.

NCCL is topology-aware and uses available hardware paths such as PCIe, NVLink, NVSwitch, and networking.

Beginner mental model:

```text
hardware paths = roads
NCCL = traffic planner for GPU communication
PyTorch distributed = user-facing API that often calls NCCL backend
```

For PyTorch:

```python
torch.distributed.init_process_group(backend="nccl")
```

Then collectives like all-reduce or all-gather often go through NCCL on NVIDIA GPUs.

## NCCL Collectives And PyTorch Distributed

PyTorch distributed is the user-facing distributed API. NCCL is the common backend for NVIDIA GPU communication.

Beginner mental model:

```text
torch.distributed = API that framework code calls
NCCL = GPU communication engine underneath
PCIe/NVLink/RDMA = physical roads NCCL can use
```

A process group is a group of ranks that communicate together:

```python
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
```

In multi-GPU inference, a rank usually maps to one GPU process.

### Collective vs P2P

A collective is an operation where all ranks in a group participate.

Examples:

```text
all-reduce
all-gather
reduce-scatter
broadcast
all-to-all
```

P2P means point-to-point communication:

```text
send
recv
isend
irecv
```

Beginner distinction:

```text
collective:
everyone in the group participates

P2P:
one rank sends to another rank
```

### All-Reduce

All-reduce reduces tensors across ranks, then gives the reduced result back to every rank.

Example:

```text
GPU0 has partial output A
GPU1 has partial output B

all-reduce(sum)

GPU0 gets A + B
GPU1 gets A + B
```

PyTorch shape:

```python
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
```

Inference use:

- tensor parallel layers often compute partial results on each GPU,
- the final activation may need the sum of those partial results,
- all-reduce is then on the critical path.

Why it matters:

```text
TP compute saving
vs
all-reduce communication cost
```

This is one reason TP=2 on `pink` improved throughput but did not reach 2x.

### All-Gather

All-gather collects tensor shards from all ranks and gives the full combined tensor to every rank.

Example:

```text
GPU0 has shard A
GPU1 has shard B

all-gather

GPU0 gets [A, B]
GPU1 gets [A, B]
```

PyTorch shape:

```python
dist.all_gather(output_tensor_list, input_tensor)
```

Inference use:

- gather sharded activations,
- gather tensor-parallel outputs when later computation needs the full tensor,
- gather sharded logits or metadata in some implementations.

All-gather is expensive when the gathered tensor is large because every rank ends up holding the full result.

### Reduce-Scatter

Reduce-scatter reduces tensors across ranks, then scatters different output shards to different ranks.

Example:

```text
GPU0 has [A0, A1]
GPU1 has [B0, B1]

reduce-scatter(sum)

GPU0 gets A0 + B0
GPU1 gets A1 + B1
```

Beginner mental model:

```text
all-reduce = reduce, then everyone gets the full result
reduce-scatter = reduce, then each rank keeps only its shard
```

Inference use:

- useful when the next computation can continue with sharded tensors,
- can reduce memory traffic compared with all-reduce plus slicing,
- appears in tensor-parallel and sequence-parallel style designs.

### All-To-All

All-to-all sends different chunks from every rank to every other rank.

Example:

```text
GPU0 sends token chunks to GPU0/GPU1/GPU2/GPU3
GPU1 sends token chunks to GPU0/GPU1/GPU2/GPU3
...
```

Inference use:

- expert parallel MoE token dispatch,
- expert output combine,
- routing tokens to the GPUs that own the selected experts.

All-to-all can be a major bottleneck in MoE inference because token routing creates irregular communication.

### P2P Send/Recv

P2P sends data from one rank to another.

PyTorch shape:

```python
dist.send(tensor, dst=next_rank)
dist.recv(tensor, src=prev_rank)
```

Inference use:

- pipeline parallel stage-to-stage activation transfer,
- prefill-decode KV cache movement in some designs,
- custom distributed cache transfer paths.

P2P is not automatically cheaper than collectives. Its cost still depends on message size, topology, overlap, and whether the data path uses PCIe, NVLink, or RDMA.

### Mapping To Inference Parallelism

```text
Tensor Parallel:
all-reduce / all-gather / reduce-scatter

Pipeline Parallel:
P2P send/recv

Expert Parallel:
all-to-all

Prefill-Decode Disaggregation:
P2P / RDMA / KV-transfer layer
```

For decode-heavy inference, these operations are especially important because decode runs one token step at a time. If every decode step triggers communication, communication latency directly affects TPOT.

## Connecting To PyTorch Distributed Primitives

The job description mentions:

- all-gather,
- reduce-scatter,
- P2P.

Inference examples:

```text
tensor parallel:
partial results across GPUs
-> all-reduce / all-gather / reduce-scatter

expert parallel:
tokens routed to experts
-> all-to-all / dispatch-combine style communication

pipeline parallel:
activations move stage to stage
-> P2P send/recv

PD disaggregation:
KV cache moves prefill worker to decode worker
-> P2P / RDMA / cache transfer layer
```

Hardware determines how expensive these primitives are.

## How To Explain `pink`

`pink` has 2 x RTX 4090.

The measured topology was:

```text
GPU0 <-> GPU1: SYS
```

Interview interpretation:

```text
The two GPUs do not have NVLink. Tensor parallel communication crosses the system/PCIe path, so TP=2 can reduce compute per GPU but pays cross-GPU communication overhead. That is why the observed speedup was meaningful but below 2x.
```

This is exactly the kind of systems reasoning an inference framework role expects.

## PD Disaggregation Connection

PD disaggregation needs KV cache transfer:

```text
prefill worker GPU
-> KV cache transfer
-> decode worker GPU
```

Possible data paths:

```text
same node, no NVLink:
GPU -> PCIe/root complex -> GPU

same node, NVLink/NVSwitch:
GPU -> NVLink/NVSwitch -> GPU

cross node, no GPUDirect RDMA:
GPU -> CPU memory -> NIC -> network -> CPU memory -> GPU

cross node, GPUDirect RDMA:
GPU -> NIC/RDMA -> network -> NIC/RDMA -> GPU
```

The better the data path, the more attractive PD disaggregation becomes.

## Interview Sentence

```text
我会把这些互联分成单机和跨机两类。PCIe 是服务器里 GPU、NIC、NVMe 等设备常用的通用互联；NVLink/NVSwitch 是 NVIDIA 面向 GPU 间通信的高带宽互联，适合 tensor parallel、expert parallel 这类频繁跨 GPU 通信。NIC 是跨机器网络入口，InfiniBand/RoCE 加 RDMA 可以降低跨节点通信 CPU 开销。GPUDirect RDMA 则进一步允许 RDMA NIC 直接读写 GPU memory，避免 GPU data 先绕到 CPU host memory。

在 LLM 推理里，这些互联会影响 TP all-reduce、PP P2P、EP all-to-all，以及 PD disaggregation 里的 KV cache transfer。比如我的双 4090 机器 GPU 拓扑是 SYS，不是 NVLink，所以 TP=2 虽然提升 throughput，但加速不到 2x，因为每层跨 GPU 通信要走较慢的系统/PCIe 路径。对于 Mooncake 或 PD 分离这类架构，KV cache 可能要跨节点传输，这时 RDMA 和 GPUDirect RDMA 是否可用，会直接影响 KV transfer time 和整体收益。
```

## Sources

- NVIDIA GPUDirect RDMA documentation: https://docs.nvidia.com/cuda/gpudirect-rdma/
- NVIDIA GPUDirect RDMA user manual: https://networking-docs.nvidia.com/gpudirectrdma/
- NVIDIA NVLink page: https://www.nvidia.com/en-us/data-center/nvlink/
- NVIDIA NCCL overview: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html
- NVIDIA NCCL GitHub: https://github.com/NVIDIA/nccl
- PyTorch distributed documentation: https://pytorch.org/docs/stable/distributed.html

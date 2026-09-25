# Handoff: compiler-driven communication pipelining

## What this project is

Take an OpenMP offload region that moves data, computes on it, and moves the
result out:

```c
#pragma omp target
for (...) {
    /* communication  */
    /* computation    */
    /* communication  */
}
```

Run coarsely, those three phases are serial and the total is their sum. Split
the data into chunks and stagger them across streams / workgroups / teams and
the total becomes roughly the longest phase plus a pipeline tail -- chunk 1's
transfer happens underneath chunk 0's compute.

**The compiler pass is what makes that rewrite safe.** Chunking a transfer and
reordering it against compute is only legal under conditions the pass has to
establish: which bytes each chunk owns, that no chunk reads data another chunk
has not delivered yet, and that the arrival of chunk *i* is observed before
anything consumes it. The performance is the reason to do it; **correctness is
the deliverable**. A pipelined kernel that races is worthless, and races here
are silent -- you get a plausible-looking wrong answer, not a crash.

This repository is the substrate: a communication library whose operations can
be issued **from inside a target region by GPU threads**, which is what makes
chunk-granularity pipelining expressible at all.

It has two network backends, and you can do this project on either:

| backend | machine | how a GPU thread moves data |
|---|---|---|
| **libfabric / CXI** | AMD MI250X + Slingshot (Tioga) | pulls a trigger that releases descriptors the host staged (DWQ), or pushes onto a ring a CPU proxy drains |
| **InfiniBand / verbs** | NVIDIA + ConnectX (e.g. MAPLE, GH200) | writes the work request itself and rings the NIC doorbell -- no staging, no trigger, no proxy |

Same header, same `ompx_*` calls, same source. §1-§4 describe Tioga, §5
InfiniBand; read §2 either way.

Nothing in this project involves machine learning. There is ML-flavoured code
elsewhere in the tree (`tools/gicc-passes/python/`); it belongs to an unrelated
effort and is not part of this work.

---

## 1. Build (Tioga)

Three things get built, in this order. All paths below assume the repository
root as the working directory. For InfiniBand see §5.2 instead.

### 1.1 Environment

```bash
export LD_LIBRARY_PATH=/opt/rocm-6.4.0/lib/llvm/lib:/opt/rocm-6.4.0/lib:/opt/cray/pe/lib64
export GICC_ROOT=$PWD
```

Export it in the shell. Do **not** pass `LD_LIBRARY_PATH` inline to `flux run`
-- that drops flux's own Cray PE paths and PMI fails at startup.

The toolchain is ROCm 6.4.0's clang (`/opt/rocm-6.4.0/lib/llvm/bin/clang++`).
This is not interchangeable: the LTO pass plugin is built against that LLVM and
will not load in a different clang.

### 1.2 The runtime library

```bash
bash omp/build_libgicc_omp.sh          # -> build_ofi/lib/libgicc_omp.{so,a}
```

Rebuild this whenever you touch anything under `src/gicc/`. It is not a
dependency-tracked build; nothing will tell you it is stale.

### 1.3 The LTO pass plugin

```bash
mkdir -p tools/gicc-passes/build && cd tools/gicc-passes/build
cmake -DCMAKE_CXX_COMPILER=/opt/rocm-6.4.0/lib/llvm/bin/clang++ ..
make -j
cd -
```

Also not dependency-tracked against the rest of the tree. A stale plugin fails
**silently**: it loads, reports `mode=passthrough`, transforms nothing, and the
build succeeds. If a pass-built binary behaves exactly like a non-pass one,
rebuild the plugin first and check the banner (see §4).

### 1.4 An application

```bash
bash examples/omp/build_giomp_example.sh examples/omp/hello_giomp.cpp build_ofi/hello_giomp
```

Or through CMake:

```cmake
find_package(gicc-omp REQUIRED)        # CMAKE_PREFIX_PATH=<root>/build_ofi/lib/cmake/gicc-omp
target_link_libraries(app PRIVATE gicc::omp)
```

Or flags, for a non-CMake build:

```bash
clang++ $(build_ofi/lib/bin/gicc-omp-config --cflags) app.cpp \
        $(build_ofi/lib/bin/gicc-omp-config --libs)
```

---

## 2. The API

One header, `gicc/omp.h`. Every entry point is `extern "C"` and prefixed
`ompx_`, so C and C++ applications use the same declarations.

```c
/* control    */ ompx_init  ompx_finalize  ompx_get_rank_num  ompx_get_num_ranks
/* memory     */ ompx_alloc  ompx_bind  ompx_free
/* movement   */ ompx_peer_ptr  ompx_put  ompx_get  ompx_get_single
/* completion */ ompx_trigger  ompx_quiet  ompx_barrier  ompx_fence
/* context    */ ompx_prepare
```

### 2.1 Memory is a symmetric heap

`ompx_init` registers one device allocation with the NIC and exchanges its
address once. `ompx_alloc` carves from that heap, so allocation is a pointer
bump with no registration and no exchange. Because every rank allocates in the
same order, **an address has the same heap offset on every rank** -- which is
why `ompx_put` takes ordinary local addresses. Your own `buf` names the peer's
copy of the same object.

```c
float* buf = ompx_alloc(n * sizeof(float));
ompx_put(peer, buf, buf, n * sizeof(float));   /* my buf -> peer's buf */
```

`ompx_bind(host_ptr, bytes)` is the other entry: it allocates on the heap,
binds it to a host pointer with `omp_target_associate_ptr`, and copies the data
in, so an application's existing `map` clauses and compute kernels keep working
unchanged. Use it when retrofitting an existing code; use `ompx_alloc` for new
buffers.

These are **collective**: every rank calls them in the same order with the same
sizes. That is what keeps the heap symmetric. `OMPX_HEAP_SIZE` (bytes, or a
`K`/`M`/`G` suffix) sizes the heap; the default is 16 GB, capped at 70% of free
device memory.

One consequence that is easy to miss: `ompx_bind` **writes** the heap. A peer's
put that arrives before the receiver's bind has run is overwritten by it. Put an
`ompx_barrier()` between the binds and the first transfer
(`examples/omp/e_omp_mapped_put.cpp` failed intermittently without it).

### 2.2 One name, host or device

`ompx_put`, `ompx_get`, `ompx_quiet`, `ompx_trigger` and the signal calls of
§2.5 are `declare target`. The same call works in ordinary code and inside
`#pragma omp target`:

```c
ompx_ctx* ctx = ompx_prepare();

#pragma omp target is_device_ptr(ctx)
{
    ompx_put(peer, dst, src, bytes);   /* issued by a GPU thread */
}
```

On the device they use a context published once by `ompx_prepare`; on the host
they call the runtime. This is the property the pipelining work needs: a
transfer can be issued **from the middle of a kernel**, at whatever chunk
boundary the pass decides on, rather than only before or after it.

One limit: only `ompx_trigger` has a C device body; the others are C++ on
the device. A C application issues transfers from the host and only calls
`ompx_trigger` on the device. C++ has all of them on both sides.

### 2.3 Completion has three distinct meanings

| call | meaning |
|---|---|
| `ompx_trigger` | release staged work to the NIC. Does **not** wait. |
| `ompx_quiet` | wait until the transfers *this rank issued* have landed |
| `ompx_fence` | `quiet` + `barrier`: everyone's transfers landed |

`ompx_put` under the DWQ transport only *stages* a descriptor; nothing moves
until a trigger is pulled. `ompx_quiet` pulls it as a side effect of waiting,
so an application only calls `ompx_trigger` explicitly when it wants the
transfer to start **earlier** than its own wait -- which for pipelining is the
normal case:

```c
ompx_put(peer, dst, src, bytes);        /* stage            */
#pragma omp target is_device_ptr(ctx)
{ ompx_trigger(); }                     /* GPU releases it  */
/* ... compute overlaps the transfer ... */
ompx_quiet();                           /* now wait         */
```

`ompx_trigger` is a no-op when no trigger is mapped, so the same source works
under the CPU proxy -- and on InfiniBand, where every put starts the moment it
is issued -- without branching.

### 2.4 Same-node peers need no transfer at all

`ompx_peer_ptr(peer, addr)` returns a same-node peer's address for the same
object, or NULL for a peer on another node and for your own rank. Pass it into a
target region with `is_device_ptr` and the thread that computes a boundary
value publishes it with one store -- the store *is* the transfer. For
intra-node pipelining this is the primitive you want; there is no NIC in the
path.

It is backed by CUDA / HIP IPC on both backends and needs **`GICC_HALO_IPC=1`**;
without it every call returns NULL and same-node traffic goes through the NIC.
With it, host-issued `ompx_put` / `ompx_get` to a same-node peer also become
device-to-device copies. `examples/omp/omp_peer_ptr.cpp` checks all of this
word for word.

A store through the pointer happens when the thread executes it. Anything that
changes the value *afterwards* must be written through the pointer too, or the
peer's copy goes stale -- see the minimod note in §4.

### 2.5 Arrival notification: put with signal

Chunk-level pipelining needs the receiver to observe chunk *i*'s arrival
without a barrier, or the stagger collapses. That is put-with-signal:

```c
void ompx_put_signal(int peer, void* dst, const void* src, size_t bytes,
                     int sig, unsigned long long value);   /* host or device */
void ompx_signal_wait(int sig, unsigned long long ge);     /* host or device */
unsigned long long ompx_signal_read(int sig);              /* host or device */
void ompx_signal_reset(int sig);                           /* host */
void ompx_stage_put_signal(int peer, void* dst, const void* src, size_t bytes,
                           int sig, unsigned long long value);   /* host, DWQ */
```

The payload lands, then `value` lands in the peer's slot `sig` (there are 64).
The flag never overtakes its payload, at any size: on CXI both writes ride one
endpoint with write-after-write ordering (`FI_ORDER_WAW`, `max_order_waw_size`
unlimited); on InfiniBand both are RDMA WRITEs on one RC queue pair, posted
behind one doorbell. A slot keeps the last value written: give each slot one
sender, make its values increase, and wait with `>=`.

Inside a kernel, this is how a team sends a chunk it has just produced:

```c
/* host, before the kernel: DWQ queues each chunk; proxy and InfiniBand ignore this */
for (int k = 0; k < K; ++k)
    ompx_stage_put_signal(peer, dst + k * C, src + k * C, C * sizeof(float), k, it + 1);

#pragma omp target teams num_teams(K) is_device_ptr(src, dst)
{
    const int k = omp_get_team_num();
    #pragma omp parallel for
    for (int i = 0; i < C; ++i) src[k * C + i] = f(k, i);    /* produce chunk k */
    ompx_put_signal(peer, dst + k * C, src + k * C, C * sizeof(float), k, it + 1);
}
```

and the receiver, on the host or in its own kernel, calls
`ompx_signal_wait(k, it + 1)` before reading chunk `k`.

What happens underneath depends on the transport:

- **DWQ.** The NIC can only run descriptors the host queued beforehand, so
  `ompx_stage_put_signal` queues the payload and the flag, and the device call
  rings slot `k`'s doorbell. Every slot has its own trigger counter, so each
  team releases only its own chunk, in whatever order the teams finish; the
  n-th device `put_signal` on a slot releases the n-th one staged there. What
  moves is what was staged -- the device call's arguments must match it.
  Generating that staging from the device call is the pass's job; it already
  does so for `ompx_put` markers, but not yet for `put_signal`.
- **CPU proxy.** The device call pushes the payload and the flag onto the ring,
  and `ompx_stage_put_signal` does nothing, so one source serves both.
- **InfiniBand.** The device call *is* the transfer: the thread writes both
  work requests and rings the doorbell, with whatever arguments it has at that
  moment. `ompx_stage_put_signal` does nothing. So on InfiniBand a chunked
  kernel needs **no host-side staging at all** -- the gap the DWQ bullet
  describes does not exist there.

Rules that are easy to break:

- One thread issues a device `put_signal`, after the threads that wrote `src`
  are synchronized (the end of a `parallel` region, or a barrier).
- Under DWQ, do not call `ompx_quiet` between staging and the kernel that
  releases the transfers: quiet waits for them, and nothing will release them.
- Under DWQ, at most 4096 `put_signal`s may be staged between two `ompx_quiet`
  calls.

`examples/omp/omp_put_signal.cpp` is the reference test (2 ranks). It checks
every word of every chunk the moment its flag is observed, with teams
deliberately finishing in reverse order, and then repeats with the flag
released *before* the payload -- a control that must fail, or the check could
not have seen a reordering. It ends with a ping-pong, host and device. The
half round trip for 8 bytes, two nodes:

| transport | host | device |
|---|---|---|
| DWQ (Tioga)        | 5.3 us  | 6.3 us  |
| proxy (Tioga)      | 4.5 us  | 10.1 us |
| InfiniBand (MAPLE) | ~45 us  | 8.2 us  |

The InfiniBand host number is slow because the host waits on a signal slot in
GPU memory with a `cudaMemcpy` per poll; the device side is the one to use.

---

## 3. Running (Tioga)

```bash
flux run -q pdebug -N2 -n16 -c 8 -g1 \
    -o mpibind=off -o cpu-affinity=per-task \
    --env=HSA_XNACK=1 \
    --env=GICC_HALO_IPC=1 --env=ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    ./your_app
```

Every flag there is load-bearing:

- **`HSA_XNACK=1`** -- without it any run with more than one rank per node
  faults with a null-pointer memory access. The in-kernel path touches
  host-mapped memory that MI250X only reaches with XNACK on.
- **`-o mpibind=off` + `ROCR_VISIBLE_DEVICES=0..7`** -- multi-node IPC needs
  every GPU on the node visible so the per-rank device select works. Do not
  also set `HIP_VISIBLE_DEVICES`; setting both breaks GPU visibility.
- **`-c 8 -o cpu-affinity=per-task`** -- `flux run` gives one core per task and
  binds nothing by default, and `mpibind=off` disables what binding there was.
  Unpinned, this benchmark varies **22% run to run** with a different straggler
  rank every iteration; pinned, **3%**. Compute is unaffected either way, so
  the whole cost lands on the communication threads. Pass both before reading
  anything into a timing.
- **Top-level `flux run`** -- nesting it inside `flux alloc` makes every rank
  report `no ROCm-capable device is detected`. If you want several measurements
  from one job, use the application's own repeat option, not a nested launch.

Transport selection:

| variable | effect |
|---|---|
| (unset) | **DWQ** -- the GPU triggers its own transfers. The default. |
| `GICC_HALO_DWQ=0` | CPU proxy: a host worker drains a ring and issues the writes |
| `GICC_SKIP_DWQ_INIT=1` | forces the proxy -- it leaves the trigger BAR unmapped, so the DWQ physically cannot fire |

---

## 4. Working on the pass

The pass plugin is `tools/gicc-passes/`, loaded with
`-fpass-plugin=libgicc-passes.so`. It is configured by **environment
variables**, not `-mllvm` flags -- those do not propagate through
`-fpass-plugin` on this clang.

```
GICC_MODE       discover | feature-extract | lower | omp-dwq | passthrough
GICC_META_DIR   directory for per-kernel JSON
```

What it does for OpenMP today (`GICC_MODE=omp-dwq`): it finds marker calls
inside `__omp_offloading_*` kernels, erases them from the device code,
synthesizes a host-side `gicc_trace_<kernel>` function that evaluates their
arguments and stages the descriptors **before the launch**, and lowers the
in-region flush to the trigger write. `examples/omp/jacobi_e2e.cpp` and
`batch_e2e.cpp` are the worked examples; `examples/omp/build_jacobi_e2e.sh`
builds both a pass and a no-pass variant of the same source so you can diff
their behaviour.

Confirm the pass actually ran:

```
[gicc-pass] mode=omp-dwq ...
[omp-dwq-discovery] kernel __omp_offloading_..._main_l108 has 3 sites
[omp-host-discovery] inserted trace call for __omp_offloading_..._main_l108 (9 captures)
```

`mode=passthrough` means the plugin is stale or `GICC_MODE` did not reach it.

It compiles the app TU **twice** with the same `GICC_META_DIR`: the host pass
needs the per-kernel JSON that the device pass writes, and on a first compile
the host side runs before that file exists.

What it does **not** do yet, and what this project is about: it does not chunk
anything. It moves whole operations. Splitting a transfer into chunks, placing
the pieces against the compute, and proving the result is equivalent to the
unchunked program is the work.

The pass exists for CXI, where device code cannot start a transfer the host did
not stage. On InfiniBand there is nothing to stage: the markers compile to plain
device puts and the same source runs without the pass (§5). A chunking
transformation that emits device `ompx_put` / `ompx_put_signal` at chunk
boundaries is therefore directly runnable on InfiniBand, which makes it a good
place to develop and check the transformation before worrying about staging.

### Checking correctness

Every benchmark here prints a whole-field checksum, and the invariant to hold
onto is that **it does not depend on how the work was decomposed**:

- `minimod` prints `FINAL field_xor`.
- `jacobi_e2e` and `batch_e2e` print a per-rank `field_hash`, identical across
  proxy and DWQ.

Compare bitwise. Do not use a tolerance -- the point is that a pipelined
version produces the *same bits* as the serial one, and a pipelining race
shows up as a plausible number, not an obviously broken one. If a checksum
moves, treat it as a correctness bug in the transformation, not as noise, and
reproduce before concluding anything: some paths here have a known intermittent
fault, so one bad run is a signal to investigate, not a verdict.

Two things learned the hard way about what to compare against:

- **The reference is a 1-rank run on the same machine and build**, not other
  rank counts. Checksums differ across GPU architectures and compilers
  (floating-point contraction), so a Tioga value means nothing on an NVIDIA
  machine and vice versa. Several rank counts agreeing with each other is not
  proof either: they can share a bug.
- **minimod's same-node path had exactly such a bug.** With `GICC_HALO_IPC=1`,
  the stencil stores a boundary cell into the same-node neighbour's copy as it
  computes it, but the source term is added *after* `target_3d` returns, so the
  neighbour's copy missed it whenever the source cell (x = nx/2) was within 4
  planes of a same-node slab edge -- i.e. an even number of ranks on one node.
  It is fixed on the `giomp` branch of `lwshanbd/minimod-tmp` (commit
  `9ed1117`, `kernel_add_source` also writes the neighbour's copy). The
  earlier claim that grid 1200 gives `85c0f82d50171bb9` for every rank count,
  transport and overlap setting predates the fix and was never compared with a
  1-rank run; 8 GPUs on one Tioga node is exactly the affected case, so
  re-check it. This is also a good example of the race class your pass must
  rule out: a value published before its last update.

On MAPLE (GH200, `build_all_ib.sh`, see §5.5), for reference:

| run | `FINAL field_xor` |
|---|---|
| `--grid 200 --nsteps 200` | `d0d77fb2ad35b538` |
| `--grid 1200 --nsteps 1000` | `57501e573f8c9f32` |

Every configuration listed in §5.6 matches these, as do the MPI and DiOMP
variants (DiOMP only where the source is not at a slab edge -- its branch
still has the halo-before-source bug the mpi branch fixed in `5d85084`).

---

## 5. InfiniBand (NVIDIA + ConnectX, no libfabric)

If you do not have Slingshot, this is the environment to use. It is a native
verbs backend (`src/gicc/platform/mlx5/`), not libfabric's verbs provider: a
GPU thread writes the RDMA work request into the send queue and rings the NIC
doorbell itself, so a device `ompx_put` starts the moment the thread issues it.

### 5.1 Requirements

- NVIDIA GPU, sm_70 or newer (tested: GH200, sm_90).
- Mellanox / NVIDIA ConnectX NIC with DevX (`mlx5dv`), rdma-core recent enough
  to have `mlx5dv_devx_umem_reg_ex`.
- `nvidia_peermem` loaded (GPU memory registration). `lsmod | grep peermem`.
- A clang with NVPTX OpenMP offload (tested: upstream clang 22.1.7, and the
  DiOMP clang 19 fork), plus CUDA 12.x (tested 12.4).
- MPI (tested: HPC-X Open MPI 4.1 and Open MPI 5.0.5).

Tested only on GH200, where the GPU reaches the NIC through Grace. On an x86
node with a PCIe-attached GPU, the GPU's store to the NIC doorbell is a PCIe
peer-to-peer write; that should work where GPUDirect RDMA works, but has not
been tried.

### 5.2 Build

The library and the examples are built by the same two scripts as on Tioga,
with `GIOMP_BACKEND=ib`. On MAPLE:

```bash
source env_maple.sh                          # CUDA 12.4 + HPC-X from NVHPC 24.5
LLVM=/shared/data1/Users/l1065028/llvm-22.1.7
export GICC_ROOT=$PWD
export GIOMP_BACKEND=ib
export GIOMP_CXX=$LLVM/bin/clang++
export GIOMP_OMP_LIBDIR=$LLVM/lib/aarch64-unknown-linux-gnu
export GIOMP_CUDA_ROOT=$CUDA_HOME
export GIOMP_MPI_ROOT=$OPAL_PREFIX           # an MPI with include/mpi.h, lib/libmpi.so
export GIOMP_OUTDIR=$PWD/build_ib/lib GIOMP_LIBDIR=$PWD/build_ib/lib
# NVHPC's cuda/ tree lacks the curand headers clang's CUDA wrapper includes:
export CPATH=$NVHPC_ROOT/math_libs/12.4/targets/sbsa-linux/include
export LD_LIBRARY_PATH=$GIOMP_OMP_LIBDIR:$LD_LIBRARY_PATH

bash omp/build_libgicc_omp.sh                                   # -> build_ib/lib/libgicc_omp.{so,a}
bash examples/omp/build_giomp_example.sh examples/omp/omp_put_signal.cpp build_ib/omp_put_signal
```

Elsewhere, point the same variables at your clang, CUDA and MPI. Only the
application TUs need `-DGICC_PLATFORM_MLX5` (the example script adds it);
without it `gicc/omp.h` selects the libfabric backend.

No pass plugin is involved, and none is needed (§4). Everything in
`examples/omp/` builds except the files that include HIP headers or DiOMP's
(`e0a`, `e0b`, `giomp_*_eval`, `giomp_fused_kernel_bench`,
`giomp_streaming_halo`).

There is also a plain CUDA API for kernels, independent of OpenMP:

```bash
mkdir -p build_ib/cmake && cd build_ib/cmake
cmake -DGICC_BACKEND=mlx5 -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
      -DMPI_CXX_COMPILER=$OPAL_PREFIX/bin/mpicxx ../..
make -j                                     # -> examples/gicc/gicc_ib_test
```

`gicc::Runtime` + `gicc::launch<kernel>(rt, grid, block, args...)`, and in the
kernel `gicc::put / get / put_nbi / put_signal / signal_wait / quiet / flush`
on `(rank, buffer index, offset)` (`src/gicc/gicc_device.cuh`).

### 5.3 Run

```bash
srun -p maple -N2 -n2 --gres=gpu:1 --mpi=pmix --mem=0 ./your_app
```

| variable | effect |
|---|---|
| `GICC_HALO_IPC=1` | map same-node peers' heaps (CUDA IPC): `ompx_peer_ptr`, and same-node host put/get as device copies |
| `GICC_IB_LANES=N` | QPs per peer (default 1). Device calls take a `lane` argument; transfers on different lanes never wait on each other's doorbells |
| `GICC_IB_QP_DEPTH=N` | send-queue slots per QP (default 1024, at most 16384) |
| `GICC_IB_DEV=mlx5_1` | pick the HCA; by default the first one with an active InfiniBand port |
| `OMPX_HEAP_SIZE` | symmetric heap size, as on Tioga |

`GICC_HALO_DWQ` / `GICC_SKIP_DWQ_INIT` do not apply. `--mem=0` matters for
minimod at grid 1200: every rank holds the whole field on the host and srun's
default memory limit OOM-kills it.

### 5.4 What is different from Tioga

| | Tioga (CXI) | InfiniBand |
|---|---|---|
| device `ompx_put` | stages (DWQ) or pushes to the proxy ring | **is** the transfer |
| `ompx_trigger`, `ompx_stage_put_signal` | release / stage | no-ops |
| host `ompx_put` | stages or pushes | posted by the CPU on a verbs QP |
| `ompx_peer_ptr` | HIP IPC, `GICC_HALO_IPC=1` | CUDA IPC, `GICC_HALO_IPC=1` |
| LTO pass | needed for the DWQ markers | not used |

On InfiniBand only, there is a burst form for many small puts:
`gicc::omp::put_nbi(ctx, peer, dst_buf, dst_off, src_buf, src_off, bytes, lane)`
(C++, buffer-index form, `gicc/platform/mlx5/gicc_omp_device.hpp`). It posts
without ringing; the next put, `gicc::omp::flush(ctx, lane)` or quiet on that
lane rings once for all of them. Use it when a thread issues a burst of small
transfers.

### 5.5 Performance, two GH200 nodes

| | this library, GPU-initiated | CUDA-aware MPI |
|---|---|---|
| 8 B latency, half round trip | 7.2 us | 2.3 us |
| bandwidth, >= 256 KB | 24.7 GB/s | 24.7 GB/s |
| 8 B messages, 64 threads over 16 lanes (`put_nbi`) | 8.2-8.8 M msg/s | 3.75 M msg/s |
| 4 KB messages, same | 22.5-23.4 GB/s | 11.9 GB/s |

So: same bandwidth; many GPU threads posting in parallel beat MPI clearly; a
single small message is slow. The floor is the NIC's own 1.55 us
(`ib_write_lat`). A GPU thread spends ~2.7 us on a put: ~1.2 us in two round
trips to the queue counters in GPU memory (reserve a slot, wait for the
previous poster) and ~1.3 us in the two system-scope fences around the
doorbell; `put_nbi` skips the fences (~1.5 us). The NIC adds ~3 us fetching the
WQE and writing the completion across the host bridge. One thread on one QP
tops out near 0.5 M msg/s -- spread concurrent senders over lanes.

minimod (`--grid 1200 --nsteps 1000`, total seconds): 2 nodes MPI 20.53,
DiOMP 20.46, GiOMP overlap-on 19.73; 3 nodes 14.73 / 15.03 / 13.28. The
overlap hides almost all communication (0.05-0.07 s left vs 1-1.9 s). At
small grids (100 or less) the per-step fence costs more than MPI's
Isend/Irecv, and GiOMP is the slowest of the three.

To reproduce the minimod numbers: `build_all_ib.sh` in `lwshanbd/minimod-tmp`
(`main`) builds all three variants with one toolchain; it expects the `mpi`,
`diomp` and `giomp` branches checked out as sibling worktrees. DiOMP needs
`OMP_DISTRIBUTED_MEM_SIZE` (e.g. `24gb` at grid 1200).

### 5.6 Tests to run first

All on two nodes unless noted; each prints PASS/FAIL or a checksum.

```bash
srun ... build_ib/cmake/examples/gicc/gicc_ib_test         # put, put_nbi, put_signal, get, host put/get
srun ... build_ib/omp_put_signal                           # ordering, incl. the must-fail control
srun ... build_ib/e1_single_put  e2_put_quiet_loop  e3_halo  e4_get  e_omp_mapped_put
srun ... build_ib/jacobi_e2e                               # field_hash 00013ad68054165d at 2 and 4 ranks
GICC_HALO_IPC=1 srun -N1 -n2 ... build_ib/omp_peer_ptr     # and -N2 -n4 --ntasks-per-node=2
```

minimod: 1, 2, 3 nodes; 1 node with 2, 3, 4 ranks under `GICC_HALO_IPC=1`;
overlap on and off -- all bit-identical to the 1-rank run (§4).

---

## 6. Traps worth knowing before you lose a day to them

- **Stale artifacts do not announce themselves.** `omp/build_libgicc_omp.sh`
  and the pass plugin are both hand-rolled builds with no dependency tracking.
  The clang-21 variants (`build_ofi/lib-clang21*`) are worse: their scripts
  rebuild **only if the file is missing**, so they silently keep serving an old
  library forever. Rebuild explicitly after changing `src/`.
- **A stale pass plugin is silent.** It reports `passthrough` and produces a
  working binary that skipped every transformation.
- **Never measure across allocations.** The same binary in the same
  configuration has been observed at 14.60 s and 11.27 s in two different
  allocations. Interleave the variants you are comparing inside one allocation,
  or submit them together; a single run per configuration proves nothing.
- **Do not put anything a job needs under `/tmp`.** It is node-local. Scripts,
  binaries and outputs go on a shared filesystem (`/p/lustre2/...` on Tioga,
  `/shared/data1/...` on MAPLE) or the job will not find them.
- **`--niters` reruns the whole simulation in one process**, which is how you
  get several measurements from one job instead of several jobs. The queue here
  is frequently deep; one job with N iterations beats N jobs.
- **`cudaMemset` on device memory can return before it is done.** Clearing a
  buffer and then letting a peer RDMA into it races: the clear can land after
  the data. Synchronize after the clear (this made `gicc_ib_test`'s host-get
  check fail intermittently, by whole pages).
- **On a node with one GPU, extra ranks share it.** MAPLE has one GH200 per
  node, so `-n4` on two nodes puts two processes on each GPU. Correctness
  tests are fine that way; timings are not. One case was seen to hang
  (`gicc_ib_test` with two ranks per GPU, in a kernel waiting on a signal the
  other process's kernel had to send) and was not investigated.
- **MAPLE specifics.** At most 3 GPUs per user in the `maple` partition
  (`QOSMaxGRESPerUser` beyond that). `salloc` needs `--account=app`; `srun`
  does not. Open MPI 5's `coll_hcoll` prints errors with two ranks per node;
  results are unaffected. A binary whose `main` never calls `MPI_Finalize`
  (the mpi branch of minimod) is killed by PMIx at exit, which also loses an
  `nsys` report.

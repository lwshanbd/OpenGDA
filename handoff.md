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

Nothing in this project involves machine learning. There is ML-flavoured code
elsewhere in the tree (`tools/gicc-passes/python/`); it belongs to an unrelated
effort and is not part of this work.

---

## 1. Build

Three things get built, in this order. All paths below assume the repository
root as the working directory.

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
sizes. That is what keeps the heap symmetric.

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
under the CPU proxy without branching.

### 2.4 Same-node peers need no transfer at all

`ompx_peer_ptr(peer, addr)` returns a same-node peer's address for the same
object, or NULL cross-node. Pass it into a target region with `is_device_ptr`
and the thread that computes a boundary value publishes it with one store --
the store *is* the transfer. For intra-node pipelining this is the primitive
you want; there is no NIC in the path.

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
Both writes ride one endpoint, and CXI grants write-after-write ordering
(`FI_ORDER_WAW`, with `max_order_waw_size` unlimited), so the flag never
overtakes its payload at any size. A slot keeps the last value written: give
each slot one sender, make its values increase, and wait with `>=`.

Inside a kernel, this is how a team sends a chunk it has just produced:

```c
/* host, before the kernel: DWQ queues each chunk; the proxy ignores this */
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

Rules that are easy to break:

- One thread issues a device `put_signal`, after the threads that wrote `src`
  are synchronized (the end of a `parallel` region, or a barrier).
- Under DWQ, do not call `ompx_quiet` between staging and the kernel that
  releases the transfers: quiet waits for them, and nothing will release them.
- At most 4096 `put_signal`s may be staged between two `ompx_quiet` calls.

`examples/omp/omp_put_signal.cpp` is the reference test (2 ranks). It checks
every word of every chunk the moment its flag is observed, with teams
deliberately finishing in reverse order, and then repeats with the flag
released *before* the payload -- a control that must fail, or the check could
not have seen a reordering. It ends with a ping-pong, host and device. On two
Tioga nodes, both transports pass, and the half round trip for 8 bytes is:

| transport | host | device |
|---|---|---|
| DWQ   | 5.3 us | 6.3 us |
| proxy | 4.5 us | 10.1 us |

---

## 3. Running

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

### Checking correctness

Every benchmark here prints a whole-field checksum, and the invariant to hold
onto is that **it does not depend on how the work was decomposed**:

- `minimod` prints `FINAL field_xor`. At grid 1200 it is
  `85c0f82d50171bb9` for every rank count, transport, and overlap setting.
- `jacobi_e2e` and `batch_e2e` print a per-rank `field_hash`, identical across
  proxy and DWQ.

Compare bitwise. Do not use a tolerance -- the point is that a pipelined
version produces the *same bits* as the serial one, and a pipelining race
shows up as a plausible number, not an obviously broken one. If a checksum
moves, treat it as a correctness bug in the transformation, not as noise, and
reproduce before concluding anything: some paths here have a known intermittent
fault, so one bad run is a signal to investigate, not a verdict.

---

## 5. If you are not on Tioga

The cross-node pipeline needs the CXI provider. Everything else degrades
gracefully, and **the intra-node work is a complete project on its own** --
figure (b) staggers intra-node communication, compute, and inter-node
communication, and the first two of those need no NIC at all.

| environment | IPC (same node) | CPU proxy | DWQ |
|---|---|---|---|
| Slingshot / CXI (Tioga) | yes | yes | yes |
| InfiniBand / verbs, through libfabric | yes | yes | **no** |
| single node, any fabric | yes | n/a | n/a |
| no libfabric at all | **does not build** | | |

- **InfiniBand / libverbs.** libfabric ships a `verbs` provider, so the library
  builds and runs. The DWQ does not work there: its trigger is a CXI-specific
  counter BAR (`fi_cxi_cntr_ops`), and there is no equivalent. Run with
  `GICC_HALO_DWQ=0` and use the CPU proxy for cross-node traffic.
- **Single node.** Every peer is same-node, so every transfer takes the IPC
  path -- `ompx_peer_ptr` plus in-kernel stores, or a device-to-device copy.
  No NIC is involved. This is enough to build and study a chunked pipeline:
  the staging, the chunk boundaries, the arrival ordering and the correctness
  argument are all the same, and you are not blocked on a fabric.
- **No libfabric.** `libgicc_omp` includes the OFI runtime directly and will
  not compile. There is an MLX5/IB backend under `src/gicc/platform/mlx5/`,
  but GiOMP is not wired to it and the LTO pass does not target it.

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
  binaries and outputs go on `/p/lustre2/...` or the job will not find them.
- **`--niters` reruns the whole simulation in one process**, which is how you
  get several measurements from one job instead of several jobs. The queue here
  is frequently deep; one job with N iterations beats N jobs.

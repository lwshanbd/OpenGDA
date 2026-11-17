// C++ spells C's restrict __restrict; the headers below use it too.
#ifdef __cplusplus
#define restrict __restrict
#endif
#include "../target_3d.h"
#include "multi_gpu.h"
#include "gicc/omp.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Built as C++ (build_maple_ib.sh), the unit includes the pipelining header,
// whose device helpers the gicc-passes plugin calls when it moves the halo
// puts after the fused kernel into that kernel (GICC_HALO_OVERLAP=2).
#ifdef __cplusplus
#include "gicc/omp_pipeline.h"
#endif

// GICC_HALO_OVERLAP=1: overlap the halo transfer with the interior compute.
// =0: original blocking halo (full-slice compute after the drain). The two paths
// must produce a bit-identical field (correctness toggle).
// =2: compute, then send: the whole slice in one fused kernel, the faces of
// the field it just wrote put right after it, and the fence at the start of
// the next step. Written as a plain serial program; in a build with put
// discovery (build_maple_ib.sh PIPE=1) the plugin sends the faces from the
// kernel as their cells are written.
//
// Unset (the default) means AUTO: overlap only when some neighbor face has to
// cross the node boundary. Overlapping is not free -- splitting the slice into
// interior + two boundary slabs costs about 0.4 s of extra compute per 1000
// steps at grid 1200 -- so it wins only when there is exposed transfer to hide.
// Measured: 2 nodes 8.886 -> 6.713 s (overlap wins 1.31x), 1 node 11.286 ->
// 11.593 s (overlap loses, comm is already just 0.24 s over xGMI).
static int halo_overlap(const void* cur, int has_left, int has_right) {
    static int forced = -2;
    if (forced == -2) {
        const char* e = getenv("GICC_HALO_OVERLAP");
        forced = e ? atoi(e) : -1;
    }
    if (forced >= 0) return forced;

    // AUTO. The decision must be GLOBAL: every step ends in a barrier, so a rank
    // that does not overlap exposes its whole transfer and becomes the critical
    // path for everyone else. Overlap iff SOME rank has a face that must leave
    // the node, i.e. a neighbour that ompx_peer_ptr cannot reach directly.
    static int automatic = -1;
    if (automatic < 0) {
        int local_remote = 0;
        if (has_left  && !ompx_peer_ptr(rank - 1, cur)) local_remote = 1;
        if (has_right && !ompx_peer_ptr(rank + 1, cur)) local_remote = 1;
        MPI_Allreduce(&local_remote, &automatic, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
    }
    return automatic;
}

// Single-GPU-per-rank GICC port of minimod.
//
// Each rank holds the FULL global field arrays (whole domain layout, identical
// on every rank), computes ONLY its x-slice [gxmin, gxmax), and exchanges the
// boundary x-faces with its rank-neighbors through GICC proxy puts. A run with
// nranks==1 computes the whole domain and is the exact single-rank reference.
//
// Compute kernels are the bare omp-target stencils from omp_offload_opt-gpu
// with the i-loop clamped to the rank's owned slice via the [x3,x4) bounds
// passed by target_3d. Arrays are full-size (global layout) but indexed through
// pointers rebased to the region start, as in the mpi branch.

#define LAP coef0*u_dev[IDX3_l(i,j,k)-uv_off] \
                +coefx[1]*(u_dev[IDX3_l(i+1,j,k)-uv_off]+u_dev[IDX3_l(i-1,j,k)-uv_off]) \
                +coefy[1]*(u_dev[IDX3_l(i,j+1,k)-uv_off]+u_dev[IDX3_l(i,j-1,k)-uv_off]) \
                +coefz[1]*(u_dev[IDX3_l(i,j,k+1)-uv_off]+u_dev[IDX3_l(i,j,k-1)-uv_off]) \
                +coefx[2]*(u_dev[IDX3_l(i+2,j,k)-uv_off]+u_dev[IDX3_l(i-2,j,k)-uv_off]) \
                +coefy[2]*(u_dev[IDX3_l(i,j+2,k)-uv_off]+u_dev[IDX3_l(i,j-2,k)-uv_off]) \
                +coefz[2]*(u_dev[IDX3_l(i,j,k+2)-uv_off]+u_dev[IDX3_l(i,j,k-2)-uv_off]) \
                +coefx[3]*(u_dev[IDX3_l(i+3,j,k)-uv_off]+u_dev[IDX3_l(i-3,j,k)-uv_off]) \
                +coefy[3]*(u_dev[IDX3_l(i,j+3,k)-uv_off]+u_dev[IDX3_l(i,j-3,k)-uv_off]) \
                +coefz[3]*(u_dev[IDX3_l(i,j,k+3)-uv_off]+u_dev[IDX3_l(i,j,k-3)-uv_off]) \
                +coefx[4]*(u_dev[IDX3_l(i+4,j,k)-uv_off]+u_dev[IDX3_l(i-4,j,k)-uv_off]) \
                +coefy[4]*(u_dev[IDX3_l(i,j+4,k)-uv_off]+u_dev[IDX3_l(i,j-4,k)-uv_off]) \
                +coefz[4]*(u_dev[IDX3_l(i,j,k+4)-uv_off]+u_dev[IDX3_l(i,j,k-4)-uv_off])

// A region kernel. With GICC_ASYNC_KERNELS the seven region kernels of one
// compute_slice are issued as deferred target tasks: they write DISJOINT parts
// of v (the six PML faces and the inner box are carved out by their x/y/z
// bounds), so they may run concurrently and are joined once at the end of the
// slice. With overlap on, a slice is issued three times per step (interior +
// two 4-plane boundary slabs), so this is 21 launches whose latency would
// otherwise serialize. Joined by the taskwait in compute_slice.
#ifdef GICC_ASYNC_KERNELS
#define REGION_LOOP _Pragma("omp target teams loop collapse(3) nowait")
#else
#define REGION_LOOP _Pragma("omp target teams loop collapse(3)")
#endif

// The same-node publication below, as a statement a cell body can take or
// leave out. Each region kernel is built twice, with and without it, and the
// variant without runs whenever neither mirror exists (always, off one node).
// Not just a skipped branch: on NVPTX the two predicated stores alone cost the
// PML kernel 10 registers (106 vs 96) and a single-GPU run 7% of its compute.
#define PUBLISH \
    if (mirror_lo && i < lo_hi)  mirror_lo[IDX3_l(i,j,k)] = out; \
    if (mirror_hi && i >= hi_lo) mirror_hi[IDX3_l(i,j,k)] = out;
#define NO_PUBLISH

// One inner cell: the plain second-order update.
#define INNER_CELL(publish) do { \
        const float lap = LAP; \
        const float out = 2.f*u_dev[IDX3_l(i,j,k)-uv_off]-v_dev[IDX3_l(i,j,k)-uv_off] \
                          +vp_dev[IDX3(i,j,k)-pv_off]*lap; \
        v_dev[IDX3_l(i,j,k)-uv_off] = out; \
        publish \
    } while (0)

// mirror_lo / mirror_hi are the left / right neighbour's IPC-mapped copy of the
// SAME field, or NULL when that neighbour is not same-node reachable. A thread
// that computes a cell in the boundary slab publishes it to the neighbour with
// one extra store at the IDENTICAL index (every rank holds the full global array
// with the same layout). That store IS the halo transfer, issued from inside the
// omp target region by the thread that produced the value -- no copy kernel, no
// proxy ring, no host involvement, and not one extra byte read.
void target_inner_3d(llint nx, llint ny, llint nz,
                     llint x3, llint x4, llint y3, llint y4, llint z3, llint z4,
                     llint lx, llint ly, llint lz,
                     const float *restrict coefx, const float *restrict coefy, const float *restrict coefz,
                     const float *restrict u, float *restrict v, const float *restrict vp,
                     float *mirror_lo, llint lo_hi, float *mirror_hi, llint hi_lo)
{
    const float coef0 = coefx[0] + coefy[0] + coefz[0];
    if (x4 <= x3) return;

    // Index through pointers rebased to the start of the region, as the mpi
    // branch does: on NVPTX the full-array form costs the inner kernel 10
    // more registers (88 vs 78) for the same addresses.
    const llint uv_off = IDX3_l(x3,-ly,-lz), pv_off = IDX3(x3,0,0);
    const float *restrict u_dev = &u[uv_off];
    const float *restrict vp_dev = &vp[pv_off];
    float *restrict v_dev = &v[uv_off];

    if (mirror_lo || mirror_hi) {
        REGION_LOOP
        for (llint i = x3; i < x4; ++i) for (llint j = y3; j < y4; ++j) for (llint k = z3; k < z4; ++k)
            INNER_CELL(PUBLISH);
    } else {
        REGION_LOOP
        for (llint i = x3; i < x4; ++i) for (llint j = y3; j < y4; ++j) for (llint k = z3; k < z4; ++k)
            INNER_CELL(NO_PUBLISH);
    }
}
#undef INNER_CELL

// One PML cell: the damped update of v, then the auxiliary field phi.
#define PML_CELL(publish) do { \
        const float lap = LAP; \
        const float out = \
            ((2.f-eta_dev[IDX3_eta1(i,j,k)-eta_off]*eta_dev[IDX3_eta1(i,j,k)-eta_off] \
            +2.f*eta_dev[IDX3_eta1(i,j,k)-eta_off])*u_dev[IDX3_l(i,j,k)-uv_off] \
            -v_dev[IDX3_l(i,j,k)-uv_off] \
            +vp_dev[IDX3(i,j,k)-pv_off]* \
            (lap+phi_dev[IDX3(i,j,k)-pv_off]))/(1.f+2.f*eta_dev[IDX3_eta1(i,j,k)-eta_off]); \
        v_dev[IDX3_l(i,j,k)-uv_off] = out; \
        publish \
        phi_dev[IDX3(i,j,k)-pv_off]=(phi_dev[IDX3(i,j,k)-pv_off]- \
              ((eta_dev[IDX3_eta1(i+1,j,k)-eta_off]-eta_dev[IDX3_eta1(i-1,j,k)-eta_off]) \
              *(u_dev[IDX3_l(i+1,j,k)-uv_off]-u_dev[IDX3_l(i-1,j,k)-uv_off])*hdx_2 \
              +(eta_dev[IDX3_eta1(i,j+1,k)-eta_off]-eta_dev[IDX3_eta1(i,j-1,k)-eta_off]) \
              *(u_dev[IDX3_l(i,j+1,k)-uv_off]-u_dev[IDX3_l(i,j-1,k)-uv_off])*hdy_2 \
              +(eta_dev[IDX3_eta1(i,j,k+1)-eta_off]-eta_dev[IDX3_eta1(i,j,k-1)-eta_off]) \
              *(u_dev[IDX3_l(i,j,k+1)-uv_off]-u_dev[IDX3_l(i,j,k-1)-uv_off])*hdz_2)) \
              /(1.f+eta_dev[IDX3_eta1(i,j,k)-eta_off]); \
    } while (0)

void target_pml_3d(llint nx, llint ny, llint nz,
                   llint x3, llint x4, llint y3, llint y4, llint z3, llint z4,
                   llint lx, llint ly, llint lz,
                   float hdx_2, float hdy_2, float hdz_2,
                   const float *restrict coefx, const float *restrict coefy, const float *restrict coefz,
                   const float *restrict u, float *restrict v, const float *restrict vp,
                   float *restrict phi, const float *restrict eta,
                   float *mirror_lo, llint lo_hi, float *mirror_hi, llint hi_lo)
{
    float coef0 = coefx[0] + coefy[0] + coefz[0];
    if (x4 <= x3) return;

    // Rebased pointers: see target_inner_3d.
    const llint uv_off = IDX3_l(x3,-ly,-lz), pv_off = IDX3(x3,0,0);
    const float *restrict u_dev = &u[uv_off];
    const float *restrict vp_dev = &vp[pv_off];
    float *restrict v_dev = &v[uv_off];
    const llint eta_off = IDX3_eta1(x3-1,-1,-1);
    const float *restrict eta_dev = &eta[eta_off];
    float *restrict phi_dev = &phi[pv_off];

    if (mirror_lo || mirror_hi) {
        REGION_LOOP
        for (llint i = x3; i < x4; ++i) for (llint j = y3; j < y4; ++j) for (llint k = z3; k < z4; ++k)
            PML_CELL(PUBLISH);
    } else {
        REGION_LOOP
        for (llint i = x3; i < x4; ++i) for (llint j = y3; j < y4; ++j) for (llint k = z3; k < z4; ++k)
            PML_CELL(NO_PUBLISH);
    }
}
#undef PML_CELL

// ---------------------------------------------------------------------------
// MINIMOD_FUSED_KERNEL: one kernel per slab instead of seven.
//
// The six PML slabs and the inner box tile [0,nx) x [0,ny) x [0,nz) exactly
// (grid.c sets x2==x3, x4==x5 and likewise for y and z), so a single loop over
// the whole box covers the same cells, with each thread selecting its formula
// from its own position. That trades seven kernel launches and their tail
// effects for one branch per cell.
// ---------------------------------------------------------------------------
// One cell of the fused kernel: inner or PML formula by position.
#define FUSED_CELL(publish) do { \
        const float lap = LAP; \
        float out; \
        if (i >= x3 && i < x4 && j >= y3 && j < y4 && k >= z3 && k < z4) { \
            out = 2.f*u_dev[IDX3_l(i,j,k)-uv_off]-v_dev[IDX3_l(i,j,k)-uv_off] \
                  +vp_dev[IDX3(i,j,k)-pv_off]*lap; \
        } else { \
            const float e = eta_dev[IDX3_eta1(i,j,k)-eta_off]; \
            out = ((2.f-e*e+2.f*e)*u_dev[IDX3_l(i,j,k)-uv_off] \
                   -v_dev[IDX3_l(i,j,k)-uv_off] \
                   +vp_dev[IDX3(i,j,k)-pv_off]*(lap+phi_dev[IDX3(i,j,k)-pv_off]))/(1.f+2.f*e); \
            phi_dev[IDX3(i,j,k)-pv_off]=(phi_dev[IDX3(i,j,k)-pv_off]- \
                  ((eta_dev[IDX3_eta1(i+1,j,k)-eta_off]-eta_dev[IDX3_eta1(i-1,j,k)-eta_off]) \
                  *(u_dev[IDX3_l(i+1,j,k)-uv_off]-u_dev[IDX3_l(i-1,j,k)-uv_off])*hdx_2 \
                  +(eta_dev[IDX3_eta1(i,j+1,k)-eta_off]-eta_dev[IDX3_eta1(i,j-1,k)-eta_off]) \
                  *(u_dev[IDX3_l(i,j+1,k)-uv_off]-u_dev[IDX3_l(i,j-1,k)-uv_off])*hdy_2 \
                  +(eta_dev[IDX3_eta1(i,j,k+1)-eta_off]-eta_dev[IDX3_eta1(i,j,k-1)-eta_off]) \
                  *(u_dev[IDX3_l(i,j,k+1)-uv_off]-u_dev[IDX3_l(i,j,k-1)-uv_off])*hdz_2)) \
                  /(1.f+e); \
        } \
        v_dev[IDX3_l(i,j,k)-uv_off] = out; \
        publish \
    } while (0)

// The fused kernel over [cxlo, cxhi), then the puts of the faces of v it has
// written (a negative peer sends nothing). Built twice, with and without the
// same-node publication, each variant in a function of its own: written as
// two branches of one function, clang merges their identical put tails into
// one block after both launches, and put discovery then finds no kernel
// that block follows.
#define FUSED_SLICE(name, publish) \
static __attribute__((noinline)) void name( \
        llint nx, llint ny, llint nz, llint x3, llint x4, llint y3, llint y4, \
        llint z3, llint z4, llint lx, llint ly, llint lz, \
        float hdx_2, float hdy_2, float hdz_2, const float *restrict coefx, \
        const float *restrict coefy, const float *restrict coefz, \
        const float *restrict u, float *restrict v, const float *restrict vp, \
        float *restrict phi, const float *restrict eta, llint cxlo, llint cxhi, \
        float *mirror_lo, llint lo_hi, float *mirror_hi, llint hi_lo, \
        int put_lo, size_t lo_off, int put_hi, size_t hi_off, size_t face_bytes) \
{ \
    const float coef0 = coefx[0] + coefy[0] + coefz[0]; \
    /* Rebased pointers: see target_inner_3d. */ \
    const llint uv_off = IDX3_l(cxlo,-ly,-lz), pv_off = IDX3(cxlo,0,0); \
    const float *restrict u_dev = &u[uv_off]; \
    const float *restrict vp_dev = &vp[pv_off]; \
    float *restrict v_dev = &v[uv_off]; \
    const llint eta_off = IDX3_eta1(cxlo-1,-1,-1); \
    const float *restrict eta_dev = &eta[eta_off]; \
    float *restrict phi_dev = &phi[pv_off]; \
    _Pragma("omp target teams loop collapse(3)") \
    for (llint i = cxlo; i < cxhi; ++i) for (llint j = 0; j < ny; ++j) for (llint k = 0; k < nz; ++k) \
        FUSED_CELL(publish); \
    if (put_lo >= 0) ompx_put(put_lo, (char*)v + lo_off, (char*)v + lo_off, face_bytes); \
    if (put_hi >= 0) ompx_put(put_hi, (char*)v + hi_off, (char*)v + hi_off, face_bytes); \
}
FUSED_SLICE(fused_slice_publish, PUBLISH)
FUSED_SLICE(fused_slice_plain, NO_PUBLISH)
#undef FUSED_SLICE

static void compute_slice_fused(llint nx, llint ny, llint nz,
                                llint x3, llint x4, llint y3, llint y4,
                                llint z3, llint z4,
                                llint lx, llint ly, llint lz,
                                float hdx_2, float hdy_2, float hdz_2,
                                const float *restrict coefx,
                                const float *restrict coefy,
                                const float *restrict coefz,
                                const float *restrict u, float *restrict v,
                                const float *restrict vp, float *restrict phi,
                                const float *restrict eta,
                                llint cxlo, llint cxhi,
                                float *mirror_lo, llint lo_hi,
                                float *mirror_hi, llint hi_lo,
                                int put_lo, size_t lo_off,
                                int put_hi, size_t hi_off, size_t face_bytes)
{
    if (cxhi <= cxlo) return;
    (mirror_lo || mirror_hi ? fused_slice_publish : fused_slice_plain)(
        nx,ny,nz, x3,x4, y3,y4, z3,z4, lx,ly,lz, hdx_2,hdy_2,hdz_2,
        coefx,coefy,coefz, u,v,vp,phi,eta, cxlo,cxhi,
        mirror_lo,lo_hi,mirror_hi,hi_lo, put_lo,lo_off,put_hi,hi_off,face_bytes);
}
#undef FUSED_CELL

// Clamp a global x-region [a,b) to the rank's owned slice [gxmin,gxmax).
static inline void clamp_x(llint a, llint b, llint gxmin, llint gxmax,
                           llint *ca, llint *cb)
{
    llint lo = a < gxmin ? gxmin : a;
    llint hi = b > gxmax ? gxmax : b;
    if (hi < lo) hi = lo;
    *ca = lo; *cb = hi;
}

// Run the full 7-region stencil sequence (the body of target_3d's compute) over
// ONLY the x-slab [cxlo, cxhi). Splitting the owned slice into an interior slab
// (halo-independent) and the two 4-plane edge slabs lets the interior be
// computed while the halo transfer is in flight. Each cell is computed exactly
// once across disjoint slabs and each cell's update reads only u (+ its own old
// v/phi), so the slab decomposition is bit-identical to the single full-slice
// compute.
static void compute_slice(llint nx, llint ny, llint nz,
                          llint x1, llint x2, llint x3, llint x4, llint x5, llint x6,
                          llint y1, llint y2, llint y3, llint y4, llint y5, llint y6,
                          llint z1, llint z2, llint z3, llint z4, llint z5, llint z6,
                          llint lx, llint ly, llint lz,
                          float hdx_2, float hdy_2, float hdz_2,
                          const float *restrict coefx, const float *restrict coefy,
                          const float *restrict coefz,
                          const float *restrict u, float *restrict v,
                          const float *restrict vp, float *restrict phi,
                          const float *restrict eta,
                          llint cxlo, llint cxhi,
                          float *mirror_lo, llint lo_hi,
                          float *mirror_hi, llint hi_lo)
{
    const llint xmin = 0; const llint xmax = nx;
    const llint ymin = 0; const llint ymax = ny;
    llint a, b;
    if (cxhi <= cxlo) return;

#ifdef MINIMOD_FUSED_KERNEL
    (void)xmin; (void)xmax; (void)ymin; (void)ymax; (void)a; (void)b;
    (void)x1; (void)x2; (void)x5; (void)x6;
    (void)y1; (void)y2; (void)y5; (void)y6;
    (void)z1; (void)z2; (void)z5; (void)z6;
    compute_slice_fused(nx,ny,nz, x3,x4, y3,y4, z3,z4, lx,ly,lz,
                        hdx_2,hdy_2,hdz_2, coefx,coefy,coefz,
                        u,v,vp,phi,eta, cxlo,cxhi,
                        mirror_lo,lo_hi,mirror_hi,hi_lo, -1, 0, -1, 0, 0);
    return;
#endif

    clamp_x(xmin, xmax, cxlo, cxhi, &a, &b);   // Damping Bottom
    target_pml_3d(nx,ny,nz, a,b, ymin,ymax, z1,z2, lx,ly,lz,
                  hdx_2,hdy_2,hdz_2, coefx,coefy,coefz, u,v,vp,phi,eta,
                  mirror_lo,lo_hi,mirror_hi,hi_lo);
    clamp_x(xmin, xmax, cxlo, cxhi, &a, &b);   // Damping Back
    target_pml_3d(nx,ny,nz, a,b, y1,y2, z3,z4, lx,ly,lz,
                  hdx_2,hdy_2,hdz_2, coefx,coefy,coefz, u,v,vp,phi,eta,
                  mirror_lo,lo_hi,mirror_hi,hi_lo);
    clamp_x(x1, x2, cxlo, cxhi, &a, &b);       // Damping Left
    target_pml_3d(nx,ny,nz, a,b, y3,y4, z3,z4, lx,ly,lz,
                  hdx_2,hdy_2,hdz_2, coefx,coefy,coefz, u,v,vp,phi,eta,
                  mirror_lo,lo_hi,mirror_hi,hi_lo);
    clamp_x(x3, x4, cxlo, cxhi, &a, &b);       // Inner
    target_inner_3d(nx,ny,nz, a,b, y3,y4, z3,z4, lx,ly,lz,
                    coefx,coefy,coefz, u,v,vp,
                    mirror_lo,lo_hi,mirror_hi,hi_lo);
    clamp_x(x5, x6, cxlo, cxhi, &a, &b);       // Damping Right
    target_pml_3d(nx,ny,nz, a,b, y3,y4, z3,z4, lx,ly,lz,
                  hdx_2,hdy_2,hdz_2, coefx,coefy,coefz, u,v,vp,phi,eta,
                  mirror_lo,lo_hi,mirror_hi,hi_lo);
    clamp_x(xmin, xmax, cxlo, cxhi, &a, &b);   // Damping Front
    target_pml_3d(nx,ny,nz, a,b, y5,y6, z3,z4, lx,ly,lz,
                  hdx_2,hdy_2,hdz_2, coefx,coefy,coefz, u,v,vp,phi,eta,
                  mirror_lo,lo_hi,mirror_hi,hi_lo);
    clamp_x(xmin, xmax, cxlo, cxhi, &a, &b);   // Damping Top
    target_pml_3d(nx,ny,nz, a,b, ymin,ymax, z5,z6, lx,ly,lz,
                  hdx_2,hdy_2,hdz_2, coefx,coefy,coefz, u,v,vp,phi,eta,
                  mirror_lo,lo_hi,mirror_hi,hi_lo);
#ifdef GICC_ASYNC_KERNELS
    // Join the deferred target tasks issued above. The slab's cells are complete
    // only after this point, so callers may read them (and the halo may publish
    // the faces) exactly as in the synchronous build.
    #pragma omp taskwait
#endif
}

// Send this step's faces of u to the neighbours that need them.
//   InfiniBand: a GPU thread posts both puts from a target region
//     (halo_dev.cpp); the transfers start as soon as it rings the doorbell.
//   libfabric: the host stages the puts and a GPU thread pulls the trigger
//     (DWQ); under the CPU proxy they are already in flight when staged.
static void issue_halo(int put_lo, int put_hi, const float *u,
                       size_t left_off, size_t right_off, size_t face_bytes)
{
    if (!put_lo && !put_hi) return;
#if defined(GICC_PLATFORM_MLX5)
    halo_put_dev((float*)u, put_lo ? rank - 1 : -1, left_off,
                 put_hi ? rank + 1 : -1, right_off, face_bytes);
#else
    if (put_lo) ompx_put(rank - 1, (char*)u + left_off,  (char*)u + left_off,  face_bytes);
    if (put_hi) ompx_put(rank + 1, (char*)u + right_off, (char*)u + right_off, face_bytes);
    fire_trigger();
#endif
}

void target_3d(const struct grid_t grid, llint nx, llint ny, llint nz,
               llint x1, llint x2, llint x3, llint x4, llint x5, llint x6,
               llint y1, llint y2, llint y3, llint y4, llint y5, llint y6,
               llint z1, llint z2, llint z3, llint z4, llint z5, llint z6,
               llint lx, llint ly, llint lz,
               float hdx_2, float hdy_2, float hdz_2,
               const float *restrict coefx, const float *restrict coefy, const float *restrict coefz,
               const float *restrict u, float *restrict v, const float *restrict vp,
               float *restrict phi, const float *restrict eta, double *time_comm, double *time_comp)
{
    struct timespec start, issued, interior_done, waited, end;
    clock_gettime(CLOCK_REALTIME, &start);

    // This rank's owned x-slice.
    llint gxmin, gxmax;
    GPU_XMIN_XMAX(rank, gxmin, gxmax);

    const int has_left  = (rank != 0);
    const int has_right = (rank != nranks_g - 1);

    // Face geometry: left 4 planes [gxmin, gxmin+4), right 4 planes [gxmax-4,
    // gxmax). Every rank holds the full global array with identical layout, so
    // the byte offset is identical on src and dst.
    const llint plane = (ny+2*ly)*(nz+2*lz);
    const size_t face_bytes = (size_t)sizeof(float) * 4 * plane;
    size_t left_off  = (size_t)IDX3_l(gxmin,   -ly, -lz) * sizeof(float);
    size_t right_off = (size_t)IDX3_l(gxmax-4, -ly, -lz) * sizeof(float);

    // The neighbours' copies of the field we are about to WRITE. On a same-node
    // neighbour ompx_peer_ptr hands back its address for the SAME object, so the
    // thread that computes a boundary cell publishes it with one extra store and
    // that face needs no transfer at all. A cross-node neighbour returns NULL.
    float* mirror_lo = has_left  ? (float*)ompx_peer_ptr(rank - 1, v) : NULL;
    float* mirror_hi = has_right ? (float*)ompx_peer_ptr(rank + 1, v) : NULL;
    const llint lo_hi = gxmin + 4;
    const llint hi_lo = gxmax - 4;

    // Only the faces that must leave the node are sent.
    const int put_lo = has_left  && !mirror_lo;
    const int put_hi = has_right && !mirror_hi;

    const int mode = halo_overlap((const void*)u, has_left, has_right);
    if (mode == 2) {
        // The faces of u were put at the end of the previous step, as its v;
        // they (and the source cell, see halo_resend_cell) must have landed
        // before the stencil reads them. The fence also keeps this step's
        // puts into the neighbours' v off until they are done reading it as
        // their u.
        // The kernel reads this unit's device context when it sends.
        static int prepared = 0;
        if (!prepared) { ompx_prepare(); prepared = 1; }
        interior_done = start;
        ompx_fence();
        clock_gettime(CLOCK_REALTIME, &waited);
        compute_slice_fused(nx,ny,nz, x3,x4, y3,y4, z3,z4, lx,ly,lz,
                            hdx_2,hdy_2,hdz_2, coefx,coefy,coefz,
                            u,v,vp,phi,eta, gxmin,gxmax,
                            mirror_lo,lo_hi,mirror_hi,hi_lo,
                            put_lo ? rank - 1 : -1, left_off,
                            put_hi ? rank + 1 : -1, right_off, face_bytes);
        clock_gettime(CLOCK_REALTIME, &end);
    } else if (mode) {
        // Start the transfers before the interior compute so they overlap it.
        issue_halo(put_lo, put_hi, u, left_off, right_off, face_bytes);
        clock_gettime(CLOCK_REALTIME, &issued);

        // INTERIOR compute, overlapping any in-flight transfer. Cells in
        // [gxmin+4, gxmax-4) read only owned u (stencil radius 4), never the
        // exchanged ghosts. No cell here is on a face, so no mirror is passed.
        const llint ixlo = gxmin + (has_left  ? 4 : 0);
        const llint ixhi = gxmax - (has_right ? 4 : 0);
        compute_slice(nx,ny,nz, x1,x2,x3,x4,x5,x6, y1,y2,y3,y4,y5,y6, z1,z2,z3,z4,z5,z6,
                      lx,ly,lz, hdx_2,hdy_2,hdz_2, coefx,coefy,coefz, u,v,vp,phi,eta,
                      ixlo, ixhi, NULL, lo_hi, NULL, hi_lo);
        clock_gettime(CLOCK_REALTIME, &interior_done);

        ompx_fence();
        clock_gettime(CLOCK_REALTIME, &waited);

        // BOUNDARY compute. These threads publish their results to the same-node
        // neighbours as they write them -- that store IS the halo transfer.
        //
        // Computing these slabs BEFORE the interior was tried, to put the
        // publication back under the interior compute: it is correct but slower
        // (8 GPU 7.693 s vs 5.897), because ranks with two same-node neighbours
        // then do twice the pre-barrier work of ranks with a cross-node side, and
        // the resulting skew costs more than the overlap gains.
        if (has_left)
            compute_slice(nx,ny,nz, x1,x2,x3,x4,x5,x6, y1,y2,y3,y4,y5,y6, z1,z2,z3,z4,z5,z6,
                          lx,ly,lz, hdx_2,hdy_2,hdz_2, coefx,coefy,coefz, u,v,vp,phi,eta,
                          gxmin, gxmin+4, mirror_lo, lo_hi, mirror_hi, hi_lo);
        if (has_right)
            compute_slice(nx,ny,nz, x1,x2,x3,x4,x5,x6, y1,y2,y3,y4,y5,y6, z1,z2,z3,z4,z5,z6,
                          lx,ly,lz, hdx_2,hdy_2,hdz_2, coefx,coefy,coefz, u,v,vp,phi,eta,
                          gxmax-4, gxmax, mirror_lo, lo_hi, mirror_hi, hi_lo);
        clock_gettime(CLOCK_REALTIME, &end);
    } else {
        // BLOCKING reference: exchange, complete, then compute the whole slice.
        // Bit-identical result; comm fully exposed. Markers are set so the shared
        // timing below attributes the wait to comm (empty "interior").
        issue_halo(put_lo, put_hi, u, left_off, right_off, face_bytes);
        clock_gettime(CLOCK_REALTIME, &issued);
        interior_done = issued;
        ompx_fence();
        clock_gettime(CLOCK_REALTIME, &waited);
        compute_slice(nx,ny,nz, x1,x2,x3,x4,x5,x6, y1,y2,y3,y4,y5,y6, z1,z2,z3,z4,z5,z6,
                      lx,ly,lz, hdx_2,hdy_2,hdz_2, coefx,coefy,coefz, u,v,vp,phi,eta,
                      gxmin, gxmax, mirror_lo, lo_hi, mirror_hi, hi_lo);
        clock_gettime(CLOCK_REALTIME, &end);
    }

    // EXPOSED comm = the reset+barrier wall time not hidden under interior
    // compute. COMPUTE = issue (tiny put-kernel) + interior + boundary.
    *time_comm += (waited.tv_sec - interior_done.tv_sec)
                + (double)(waited.tv_nsec - interior_done.tv_nsec) / 1.0e9;
    *time_comp += (interior_done.tv_sec - start.tv_sec)
                + (double)(interior_done.tv_nsec - start.tv_nsec) / 1.0e9
                + (end.tv_sec - waited.tv_sec)
                + (double)(end.tv_nsec - waited.tv_nsec) / 1.0e9;
}

// The source is added to v after target_3d returns. Under GICC_HALO_OVERLAP=2
// the faces of v have been put by then, so kernel_add_source calls this for a
// source cell c in the left (to_lo) or right (to_hi) face: a neighbour across
// the node gets the cell again, once the face's put has completed (two RDMA
// writes to one address are not ordered otherwise). A same-node neighbour's
// copy kernel_add_source writes itself, and the other modes send the faces
// at the start of the next step anyway.
void halo_resend_cell(const float *v, llint c, int to_lo, int to_hi)
{
    if (halo_overlap((const void*)v, rank != 0, rank != nranks_g - 1) != 2) return;
    const int peers[2] = { to_lo ? rank - 1 : -1, to_hi ? rank + 1 : -1 };
    char* cell = (char*)v + c * sizeof(float);
    int quiet = 0;
    for (int s = 0; s < 2; ++s) {
        const int p = peers[s];
        if (p < 0 || p >= nranks_g || ompx_peer_ptr(p, v)) continue;
        if (!quiet) { ompx_quiet(); quiet = 1; }
        ompx_put(p, cell, cell, sizeof(float));
    }
}

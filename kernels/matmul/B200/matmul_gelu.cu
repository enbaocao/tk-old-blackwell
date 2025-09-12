#include "kittens.cuh" // ThunderKittens core: tiles, tensor memory accelerator (TMA), warp/cluster utils
#include "prototype.cuh" // TK prototype helpers: phase-bit ring, allocators, semaphores
#include <iostream> // host-side I/O for benchmarking/logging

constexpr int NUM_CONSUMERS = (2); // number of consumer warpgroups that postprocess/store results
constexpr int NUM_PRODUCERS = (1); // number of producer warpgroups that load/compute

using namespace kittens; // bring TK names into scope

static constexpr int Mb = 128; // tile size along M (rows of A / C) per cluster task
static constexpr int Nb = 256; // tile size along N (cols of B / C) per cluster task
static constexpr int Kb = 64;  // tile depth along K per iteration of the pipeline

// Fast tanh for GELU epilogue
__device__ static inline float fast_tanh(float x) {
    #if defined(__CUDA_ARCH__)
        #if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 750)
            float y;
            asm volatile ( "tanh.approx.f32 %0, %1; " : "=f"(y) : "f"(x));
            return y;
        #else
            return ::tanhf(x);
        #endif
    #else
    return std::tanh(x);
    #endif
}

template<int R, int C, kittens::ducks::rt_layout::all L>
__device__ static inline void apply_gelu_inplace(rt_fl<R,C,L> &t) {
    #pragma unroll
    for(int n = 0; n < t.height; n++) {
        #pragma unroll
        for(int m = 0; m < t.width; m++) {
            #pragma unroll
            for(int k = 0; k < t.packed_per_tile; k++) {
                float2 v = t.tiles[n][m].data[k];
                float fx = v.x; float fy = v.y;
                float zx = 0.79788456f * (fx + 0.044715f * fx * fx * fx);
                float zy = 0.79788456f * (fy + 0.044715f * fy * fy * fy);
                t.tiles[n][m].data[k].x = 0.5f * fx * (1.0f + fast_tanh(zx));
                t.tiles[n][m].data[k].y = 0.5f * fy * (1.0f + fast_tanh(zy));
            }
        }
    }
}

struct matmul_globals { // kernel parameter bundle: typed views to global memory tensors
    using a_tile = st_bf<Mb, Kb>;     // shared-memory tile for A (bf16, Mb x Kb)
    using b_tile = st_bf<Nb/2, Kb>;   // shared-memory tile for half of B (bf16, (Nb/2) x Kb)
    using d_tile = st_bf<Mb, 64>;     // shared-memory tile to stage D (bf16, Mb x 64)

    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>; // global-memory layout descriptor for A
    using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>; // global-memory layout descriptor for B
    using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>; // global-memory layout descriptor for D

    a_gl a; // A matrix in global memory
    b_gl b; // B matrix in global memory
    d_gl d; // D (output) matrix in global memory
};

constexpr int NUM_WORKERS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4; // 4 warps per warpgroup
constexpr int NUM_THREADS = NUM_WORKERS * kittens::WARP_THREADS;  // total threads per CTA

__device__ static inline int get_iters_per_task(const matmul_globals &g) { // K-loop iters for one tile task
    return g.a.cols() / Kb; // number of Kb-wide slices that cover K
}
template<int SUPER_M=8> __device__ static inline int2 get_task_idx(const matmul_globals &g, int task_iter, bool is_consumer) { // map task_iter + cluster rank to (tile-row, tile-col)
    constexpr int CLUSTER_M = 4*Mb, CLUSTER_N = Nb; // cluster tile footprint in MxN
    int cluster_x = clusterIdx().x, ctarank = cluster_ctarank(); // cluster coords and rank within cluster
    int task_id = task_iter * (gridDim.x/2) + cluster_x; // linear task id across grid halves
    int Rblocks = g.d.rows() / CLUSTER_M, Cblocks = g.d.cols() / CLUSTER_N; // number of cluster tiles along M/N
    int super_rows = (Rblocks/SUPER_M)*SUPER_M, // full SUPER_M-row groups
        final_rows = Rblocks - super_rows,      // remainder rows not filling SUPER_M
        super_repeat = SUPER_M*Cblocks;         // tiles in one SUPER_M stripe across N
    if (task_id < super_rows * Cblocks) { // within full SUPER_M stripes
        return { 
            (SUPER_M*(task_id/super_repeat) + task_id%SUPER_M)*4 + ctarank*2 + is_consumer*(warpgroup::groupid()), // M tile index adjusted by ctarank and role
            is_consumer ? (task_id%super_repeat)/SUPER_M : 2*((task_id%super_repeat)/SUPER_M) + ctarank // N tile index differs for producer/consumer
        };
    }
    else if (task_id < Rblocks*Cblocks) { // within tail rows
        int remainder_id = task_id - super_rows*Cblocks; // index into remaining row tiles
        return {
            (super_rows + remainder_id%final_rows)*4 + ctarank*2 + is_consumer*(warpgroup::groupid()), // M tile index in tail
            is_consumer ? remainder_id/final_rows : 2*(remainder_id/final_rows) + ctarank // N tile index in tail
        };
    }
    else { // out of work
        return { -1, -1 };
    }
}

__global__ __cluster_dims__(2) __launch_bounds__(NUM_THREADS, 1)
void matmul_gelu(const __grid_constant__ matmul_globals g) { // clustered kernel with 2 CTAs per cluster, fixed threads per CTA

    /*
     * High-level overview
     * - This is a clustered GEMM (2 CTAs per cluster) with one producer warpgroup and two consumer warpgroups:
     *   - Producer WG (warpgroupid == NUM_CONSUMERS) overlaps: TMA loads of A/B → MMA compute into tensor-memory accumulators.
     *   - Consumer WGs (warpgroupid in [0, NUM_CONSUMERS)) drain the accumulators, apply GELU epilogue in float, and commit results to global D via TMA stores.
     * - Inputs (A/B) are ring-buffered in shared memory with PIPE_DEPTH slots; per-slot semaphores + phase bits coordinate safe reuse.
     * - Outputs use a parity (task_iter % 2) double buffer in tensor memory so stores for tile t overlap compute for tile t+1.
     */

    extern __shared__ int __shm[];  // dynamic shared memory buffer (raw)
    tma_swizzle_allocator al((int*)&__shm[0]); // allocator that honors TMA swizzle patterns
    int warpid = kittens::warpid(), warpgroupid = warpgroup::groupid(); // intra-CTA warp id and warpgroup id
    int iters_per_task = get_iters_per_task(g); // K-iterations per tile task

    constexpr int PIPE_DEPTH = 4; // number of in-flight K-tiles in the software pipeline (ring size)
    // oust (stray token) — comment out to avoid compile error
    // oust
    using a_tile = matmul_globals::a_tile; // shared tile alias for A
    using b_tile = matmul_globals::b_tile; // shared tile alias for B
    using d_tile = matmul_globals::d_tile; // shared tile alias for D staging
    
    a_tile (&a_smem)[PIPE_DEPTH][NUM_CONSUMERS] = al.allocate<a_tile, PIPE_DEPTH, NUM_CONSUMERS>(); // ring-buffered A tiles per consumer
    b_tile (&b_smem)[PIPE_DEPTH]                = al.allocate<b_tile, PIPE_DEPTH>();                 // ring-buffered B tiles shared across consumers
    d_tile (&d_smem)                            = al.allocate<d_tile>();                             // staging tile for stores

    tma::cluster::sync(); // ensure all CTAs in cluster have allocated shared memory
    tensor_allocator<1, 2> tm_alloc{}; // allocator for tensor memory (global/cluster shared region)
    using d_tt_t = tt<float, Mb, Nb>; // tensor tile (float accumulators) of size Mb x Nb

    __shared__ kittens::semaphore inputs_arrived[PIPE_DEPTH], inputs_finished[PIPE_DEPTH], outputs_arrived, outputs_finished[NUM_CONSUMERS]; // cluster-coop semaphores
    uint32_t bitfield = 0xFFFF0000; // phase bits: finished start at 1, arrived start at 0 for ring indices

    /*
     * Semaphores + phase bits
     * - inputs_arrived[ring]: signaled by the loader when A/B have been placed into shared; compute waits on it.
     * - inputs_finished[ring]: signaled by compute after consuming a slot; loader waits on it before refilling the slot.
     * - outputs_arrived: signaled by the producer to indicate accumulators for a tile are ready for consumers.
     * - outputs_finished[c]: signaled by consumer warpgroup c to free its tensor-memory buffer for the next tile (double-buffered by parity).
     * - bitfield stores 1-bit phases per ring index for both arrived and finished domains to avoid ABA slot reuse races.
     */

    if (threadIdx.x == 0) { // one thread initializes semaphores
        for(int i = 0; i < PIPE_DEPTH; i++) { // per-ring slot semaphores
            init_semaphore(inputs_arrived[i], 0, 2);         // track A/B loads into shared
            init_semaphore(inputs_finished[i], 0, NUM_CONSUMERS); // track compute usage of that slot
        }
        init_semaphore(outputs_arrived, 0, 1); // signal that accumulators are ready to be read/stored
        for(int i = 0; i < NUM_CONSUMERS; i++) { // one per consumer warpgroup
            init_semaphore(outputs_finished[i], 0, 2); // signal that output buffer is free for next write
        }
    }

    tma::cluster::sync(); // all semaphores initialized before proceeding
    
    if(warpgroupid == NUM_CONSUMERS) { // producer warpgroup: loads A/B tiles and performs MMA to fill accumulators
        warpgroup::decrease_registers<56>(); // trade registers for occupancy where possible
        int ctarank = cluster_ctarank(); // which CTA within the cluster
        if(warpgroup::warpid() == 3) { // dedicated loader warp for TMA transfers
            /*
             * Producer loader warp
             * - Walks the task stream and refills the next ring slot per K-slice with:
             *   A[row 0], A[row 1], and the B panel corresponding to this (M,N,Kb) tile.
             * - Before writing a slot, it waits on inputs_finished[slot] so consumers/computes are done with it.
             * - On boundaries (tail of PIPE_DEPTH or prior task parity), it signals outputs_arrived to let consumers start draining.
             */
            int input_ring = 0; // which ring slot to fill next
            for(int task_iter = 0; true; task_iter++) { // iterate over assigned (M,N) tile tasks
                int2 rowcol = get_task_idx(g, task_iter, false); // compute task coordinates for producer role
                if(rowcol.x == -1) { // no more work; drain pipeline
                    for(int idx = 0; idx < (PIPE_DEPTH); idx++) {
                        tma::cluster::wait(inputs_finished[input_ring], prototype::get_phasebit<1>(bitfield, input_ring)); // wait until consumers finished with slot
                        input_ring=prototype::ring_advance<PIPE_DEPTH>(input_ring); // advance ring
                    }
                    if(laneid() == 0) arrive(outputs_arrived); // final signal that outputs are ready
                    break; // exit
                }
                for (int idx = 0; idx < iters_per_task; idx++) { // loop over K-tiles
                    tma::cluster::wait(inputs_finished[input_ring], prototype::get_phasebit<1>(bitfield, input_ring)); // wait for slot free
                    prototype::update_phasebit<1>(bitfield, input_ring); // toggle finished phase for this slot
                    if(task_iter>0 && idx==PIPE_DEPTH-1 && laneid() == 0) arrive(outputs_arrived); // overlap: signal availability for previous task
                    tma::cluster::expect(inputs_arrived[input_ring], 0, a_smem[0][0], a_smem[0][1], b_smem[0]); // announce incoming loads to sema
                    tma::cluster::load_async(a_smem[input_ring][0], g.a, {(rowcol.x+0), idx}, inputs_arrived[input_ring], (uint16_t)(1<<ctarank), 0); // load A row 0
                    tma::cluster::load_async(a_smem[input_ring][1], g.a, {(rowcol.x+1), idx}, inputs_arrived[input_ring], (uint16_t)(1<<ctarank), 0); // load A row 1
                    tma::cluster::load_async(b_smem[input_ring],    g.b, { rowcol.y,    idx}, inputs_arrived[input_ring], (uint16_t)(1<<ctarank), 0); // load B panel
                    input_ring=prototype::ring_advance<PIPE_DEPTH>(input_ring); // proceed to next ring slot
                }
            }
        }
        else if(ctarank == 0 && (warpgroup::warpid() == 0 || warpgroup::warpid() == 1)) { // compute warps: launch MMA
            /*
             * Producer compute warps (warps 0 and 1)
             * - Each warp consumes one of the A rows and the shared B panel, writing into its tensor-memory accumulator tile.
             * - First K-slice uses mm2_ABt(...) to initialize accumulators; subsequent slices use mma2_ABt(...) to accumulate.
             * - For each ring slot: wait inputs_arrived phase → flip phase → compute → advance.
             * - outputs_finished[warp] enforces a double-buffer contract with consumers; we must not overwrite an accumulator
             *   tile until the corresponding consumer parity has finished storing it.
             */
            using d_tt_t_local = d_tt_t;
            d_tt_t_local d_tt = tm_alloc.allocate<d_tt_t_local>(warpgroup::warpid()*Nb); // allocate accumulator tile in tensor memory
            int input_ring = 0; // read head of inputs
            for(int task_iter = 0; true; task_iter++) { // per (M,N) tile
                int2 rowcol = get_task_idx(g, task_iter, false); // obtain tile coords
                if(rowcol.x == -1) break; // done
                tma::cluster::wait(outputs_finished[warpgroup::warpid()], (task_iter+1)%2); // ensure output buffer is free for this warp
                tma::cluster::wait(inputs_arrived[input_ring], prototype::get_phasebit<0>(bitfield, input_ring)); // wait data ready
                prototype::update_phasebit<0>(bitfield, input_ring); // toggle arrived phase
                mm2_ABt(d_tt, a_smem[0][warpgroup::warpid()], b_smem[0], inputs_finished[0]); // first K-slice: init accumulators
                input_ring=prototype::ring_advance<PIPE_DEPTH>(input_ring); // advance
                for(int idx = 1; idx < iters_per_task; idx++) { // remaining K-slices
                    tma::cluster::wait(inputs_arrived[input_ring], prototype::get_phasebit<0>(bitfield, input_ring)); // wait next data ready
                    prototype::update_phasebit<0>(bitfield, input_ring); // update phase
                    mma2_ABt(d_tt, a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]); // accumulate with next slices
                    input_ring=prototype::ring_advance<PIPE_DEPTH>(input_ring); // next
                }
            }
        }
    }
    else { // consumer warpgroups: read accumulators, apply GELU, and commit to global D
        /*
         * Consumers
         * - Wait on outputs_arrived parity for tile t, then stream the Mb×Nb accumulator tile from tensor memory in 64‑column strips.
         * - Each strip: load float into registers (rt_fl) → apply GELU → stage into shared (d_smem as bf16) → issue TMA store_async to global D.
         * - group<8>::sync codes coordinate the two consumer warpgroups so they ping‑pong strips without clobbering staging buffers.
         * - Immediately after the first loads, consumers arrive(outputs_finished[wg]) to free the producer’s parity buffer for tile t+1.
         */
        warpgroup::increase_registers<224>(); // give consumers more registers for staging
        using d_tt_t_local = d_tt_t;
        d_tt_t_local d_tt = tm_alloc.allocate<d_tt_t_local>(warpgroupid*Nb); // their view into tensor memory accumulators
        for(int task_iter = 0; true; task_iter++) { // for each tile assignment
            int2 rowcol = get_task_idx(g, task_iter, true); // tile coords for consumers
            if(rowcol.x == -1) break; // done
            kittens::wait(outputs_arrived, task_iter%2); // wait until producers signaled availability for this parity
            rt_fl<Mb/4, d_tile::cols> d_reg[4]; // float register tiles to fetch strips of d_tt
            if(warpgroupid == 1) group<8>::sync(15); // fine-grained inter-warpgroup sync for ordering
            #pragma unroll
            for(int i = 0; i < Nb/d_tile::cols; i++) { // iterate 64-wide columns across Nb
                warpgroup::load_async(d_reg[i], d_tt.subtile<tt<float, 128, 64>>(0, 64*i)); // async load subtile into float registers
            }
            tm_load_wait(); // wait for tensor memory loads to finish
            warpgroup::sync(warpgroupid); // ensure all lanes ready
            if(warpgroup::laneid() == 0) tma::cluster::arrive(outputs_finished[warpgroupid], 0); // free producer output buffer for this warpgroup

            // GELU epilogue in float
            #pragma unroll
            for(int i = 0; i < Nb/d_tile::cols; i++) {
                apply_gelu_inplace(d_reg[i]);
            }

            if(warpgroupid == 0) group<8>::sync(15); // coordinate store order
            if(warpgroupid == 1) group<8>::sync(14); // coordinate store order
            warpgroup::store(d_smem, d_reg[0]); // stage first strip to shared for TMA store (bf16 conversion on store)
            warpgroup::sync(warpgroupid); // wait staging complete
            if(warpgroup::warpid() == 0) tma::store_async(g.d, d_smem, {rowcol.x, 4*rowcol.y+0}); // async store first strip to D
            #pragma unroll
            for(int i = 1; i < Nb/d_tile::cols; i++) { // remaining strips
                tma::store_async_read_wait(); // ensure previous store consumed staging buffer
                warpgroup::sync(warpgroupid);
                warpgroup::store(d_smem, d_reg[i]); // stage next strip
                warpgroup::sync(warpgroupid);
                if(warpgroup::warpid() == 0) tma::store_async(g.d, d_smem, {rowcol.x, 4*rowcol.y+i}); // async store next strip
            }
            tma::store_async_read_wait(); // flush last async store
            if(warpgroupid == 0) group<8>::sync(14); // align warpgroups before next tile
            group<8>::sync(15); // all consumers sync here
        }
    }
    tma::cluster::sync(); // final cluster-wide barrier before kernel exit
}


constexpr bool NCU = false; // set to false for warmup and repeated timing; true for single-run (e.g., when profiling)
constexpr bool DO_CPU_REF = true; // compute CPU reference when problem size is small enough
#include <iostream> // host I/O again for standalone main
#include <random>   // RNG for input initialization
#include <cuda_bf16.h> // CUDA bfloat16 conversions (host/device)
#include <omp.h>    // OpenMP for CPU reference
#include <math.h>   // tanh on host

// CPU reference GEMM + GELU epilogue: C = GELU(A * B^T)
void cpu_gemm(float* a, float* b, float* c, int M, int N, int K) { // simple CPU reference GEMM with GELU epilogue
    #pragma omp parallel for collapse(2) // parallelize the i,j loops for speed
    for (int i = 0; i < M; i++) { // rows of A/C
        for (int j = 0; j < N; j++) { // cols of B/C
            float sum = 0.0f; // accumulator
            for (int k = 0; k < K; k++) { // inner product across K
                sum += a[i * K + k] * b[j * N + k]; // multiply-accumulate (note: b indexed as if laid out N x K transposed)
            }
            float x = sum;
            float z = 0.79788456f * (x + 0.044715f * x * x * x);
            float gelu = 0.5f * x * (1.0f + tanhf(z));
            c[i * N + j] = gelu; // write C(i,j)
        }
    }
}

void inner_run(bf16 *d_A, bf16 *d_B, bf16 *d_C, size_t M, size_t N, size_t K, dim3 grid, dim3 block) { // one kernel launch wrapper
    using globals  = matmul_globals; // alias
    typename globals::a_gl Ag{d_A, nullptr, nullptr, M, K}; // bind A pointer and shape to TK global descriptor
    typename globals::b_gl Bg{d_B, nullptr, nullptr, N, K}; // bind B pointer and shape
    typename globals::d_gl Dg{d_C, nullptr, nullptr, M, N}; // bind D pointer and shape
    globals G{Ag, Bg, Dg}; // bundle
    matmul_gelu<<<grid, block, MAX_SHARED_MEMORY-1024>>>(G); // launch with near-max dynamic shared memory (leave headroom)
}

int run_benchmark(size_t M, size_t N, size_t K) { // allocate, initialize, run kernel, validate, and report perf
    cudaError_t cudaStatus; // for error checks

    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n"; // banner
    std::cout << "Block size: " << Mb*2 << "x" << Nb<< "\n"; // logical cluster tile size processed per task

    // Allocate host memory
    float *h_A = new float[M * K]; // host A (fp32) row-major MxK
    float *h_B = new float[K * N]; // host B (fp32) row-major KxN
    float *h_C = new float[M * N]; // host C (fp32) output MxN
    float *h_C_ref = new float[M * N]; // host reference C

    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd; // entropy
    std::mt19937 gen(42); // fixed seed for repeatability
    std::uniform_real_distribution<> dis(-0.5, 0.5); // input range

    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen); // random A
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen); // random B

    std::cout << "Initialized matrices" << std::endl;

    // Perform CPU matrix multiplication for reference with GELU epilogue (only for small sizes)
    bool do_ref = DO_CPU_REF && (M <= 2048 && N <= 2048 && K <= 2048);
    if(do_ref) cpu_gemm(h_A, h_B, h_C_ref, M, N, K); // compute reference C on CPU

    std::cout << "Performed CPU matrix multiplication + GELU (reference)" << std::endl;

    // Allocate device memory
    __nv_bfloat16 *d_A, *d_B, *d_C; // device buffers in bf16
    cudaMalloc(&d_A, M*K*sizeof(__nv_bfloat16)); // allocate A on device
    cudaMalloc(&d_B, K*N*sizeof(__nv_bfloat16)); // allocate B on device
    cudaMalloc(&d_C, M*N*sizeof(__nv_bfloat16)); // allocate C on device

    // Check for CUDA errors
    cudaStatus = cudaGetLastError(); // check for allocation errors
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl; // report
        return -1; // early out on error
    }

    std::cout << "Allocated device memory" << std::endl;

    // Convert to __nv_bfloat16 and copy to device
    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[M * K]; // temporary host bf16 buffers
    __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[K * N];
    for (int i = 0; i < M * K; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]); // convert A to bf16
    for (int i = 0; i < K * N; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]); // convert B to bf16

    cudaMemcpy(d_A, h_A_bf16, M*K*2, cudaMemcpyHostToDevice); // copy A to device
    cudaMemcpy(d_B, h_B_bf16, K*N*2, cudaMemcpyHostToDevice); // copy B to device

    std::cout << "Copied matrices to device" << std::endl;

    unsigned long mem_size = MAX_SHARED_MEMORY - 1024; // leave some headroom for system usage
    cudaFuncSetAttribute(matmul_gelu, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size); // opt-in to large dynamic smem

    // Launch kernel
    dim3 grid(148, 1); // grid size chosen to saturate device for target sizes
    dim3 block(NUM_THREADS); // single CTA per cluster with all threads
    std::cout << "Launching warmup kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n"; // warmup log
    for(int i = 0; i < (NCU ? 0 : 1); i++) { // optional warmup when not profiling
        inner_run(d_A, d_B, d_C, M, N, K, grid, block); // launch
    }

    // Start timing
    cudaDeviceSynchronize(); // sync before timing
    std::cout << "Launching kernel with grid (" << grid.x << ", " << grid.y << "), block (" << block.x << ")\n"; // timing run log
    auto start = std::chrono::high_resolution_clock::now(); // start timer

    constexpr int ITERS = (NCU ? 1 : 5); // number of timed iterations
    for(int i = 0; i < ITERS; i++) {
        inner_run(d_A, d_B, d_C, M, N, K, grid, block); // timed launches
    }
    cudaDeviceSynchronize(); // stop timer after sync

    // End timing
    auto end = std::chrono::high_resolution_clock::now(); // end timer

    // Calculate duration
    std::chrono::duration<double> diff = end - start; // elapsed seconds
    double useconds = diff.count() * 1e6 / ITERS; // average microseconds per iteration

    // Calculate TFLOPs
    double flops = double(2.0) * M * N * K; // 2 FLOPs per MAC
    double tflops = (flops / useconds) / 1e6; // convert to TFLOPs

    std::cout << "Avg Kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";
    
    // Check for CUDA errors
    cudaStatus = cudaGetLastError(); // check for runtime errors
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl; // report
        return -1; // abort on error
    }

    // Copy result back to host
    __nv_bfloat16 *h_C_bf16 = new __nv_bfloat16[M * N]; // host buffer for C in bf16
    cudaMemcpy(h_C_bf16, d_C, M*N*2, cudaMemcpyDeviceToHost); // copy back

    std::cout << "Copied result back to host" << std::endl;

    // Convert result back to float for comparison
    for (int i = 0; i < M * N; ++i) h_C[i] = __bfloat162float(h_C_bf16[i]); // convert to fp32

    std::cout << "Converted result back to float" << std::endl;

    // Verify GELU epilogue correctness against CPU reference
    if(do_ref) {
        float max_error = 0.0f; // track worst-case abs error
        float average_error = 0.0f; // accumulate average error
        int error_count = 0; // count large-error elements
        for (int i = 0; i < M * N; ++i) { // check all elements
            float error = std::abs(h_C[i] - h_C_ref[i]); // absolute difference
            if(error > 1.0) { // tolerant threshold due to bf16 rounding
                if(error_count < 20) std::cout << "Error at row " << i / N << " col " << i % N << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl; // print first few
                else if(error_count == 21) std::cout << "Too many errors to show them all.\n"; // suppress spam
                error_count++;
            }
            max_error = std::max(max_error, error); // update worst
            average_error += error; // sum for mean
        }
        average_error /= M*N; // compute mean error

        std::cout << "[VERIFY] GELU epilogue — Max error: " << max_error << std::endl;
        std::cout << "[VERIFY] GELU epilogue — Average error: " << average_error << std::endl;
        std::cout << "[VERIFY] GELU epilogue — Error count: " << error_count << std::endl;
    } else {
        std::cout << "[VERIFY] Skipping CPU reference/verification for large sizes (> " << 2048 << ")" << std::endl;
    }

    // Clean up
    delete[] h_A; // free host buffers
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_A_bf16;
    delete[] h_B_bf16;
    delete[] h_C_bf16;
    cudaFree(d_A); // free device buffers
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

int main() { // standalone driver
    int N;
    // A quick small verification run first
    N = 1024;
    run_benchmark(N, N, N);
    // Larger tests (comment out if CPU reference is too slow for your machine)
    N = 8192; run_benchmark(N, N, N);
    // N = 16384; run_benchmark(N, N, N);
    return 0; // done
}



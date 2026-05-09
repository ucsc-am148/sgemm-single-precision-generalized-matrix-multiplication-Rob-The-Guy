"""Student kernels for the SGEMM autograder assignment.

You implement K2 (GMEM coalescing), K3 (shared-memory blocking), K4 (1D
register tiling), and K5 (2D register tiling) inside this file. The launch
wrappers, tile-size constants, and signatures are provided — you only edit
the kernel bodies marked TODO.

K1 (naive) is given as a worked example so you have a reference for the
numba.cuda @cuda.jit signature every kernel must match.

To check correctness locally before submitting:
    python sanity_check.py

To submit: push your edits to the main branch of this assignment repo.
Each push that touches kernels.py triggers the autograder, which runs
on a Modal A100 40GB and posts your grade as a comment on the commit.
You have 5 graded submissions per assignment.
"""
import math

from numba import cuda, float32


# ── Tile constants ──────────────────────────────────────────────────
# These are tied to the launch shapes the autograder will use. Do not
# change them; the run_kN wrappers below depend on these values.

BLOCKSIZE = 32          # K1 + K2 tile

# K3 tile sizes
BM3, BN3, BK3 = 32, 32, 32

# K4 tile sizes
BM4, BN4, BK4 = 64, 64, 8
TM4 = 8

# K5 tile sizes
BM5, BN5, BK5 = 128, 128, 8
TM5, TN5 = 8, 8


# ── K1: naive (worked example, do not edit) ─────────────────────────

@cuda.jit
def sgemm_naive(A, B, C, M, N, K):
    """K1: one thread per output element. No tiling, no shared memory.
    Provided so you have a working numba.cuda kernel for reference.
    """
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x < M and y < N:
        tmp = float32(0.0)
        for i in range(K):
            tmp += A[x, i] * B[i, y]
        C[x, y] = tmp


# ── K2: GMEM coalescing (TODO) ──────────────────────────────────────

@cuda.jit
def sgemm_coalesced(A, B, C, M, N, K):
    """K2: rewrite K1 so that 32 threads in a warp end up writing to 32
    *consecutive columns* of C (and reading 32 consecutive elements of B).
    The arithmetic is identical to K1

    Launch shape (run_k2 below uses this):
        block = (BLOCKSIZE * BLOCKSIZE,)        # 1024 threads, 1D
        grid  = (ceil(M / BLOCKSIZE), ceil(N / BLOCKSIZE))

    With a 1D block of 1024 threads, threadIdx.x runs 0..1023.
    Derive (row_in_tile, col_in_tile) from threadIdx.x using integer division
    and modulo by BLOCKSIZE. 
    Be careful which one indexes the column.
    """
    
# 1) 1D thread id in [0, 1023]
    tid = cuda.threadIdx.x

    # 2) Map tid to a (tile_row, tile_col) in a 32x32 tile
    #    tile_row in [0,31], tile_col in [0,31]
    tile_row = tid // BLOCKSIZE
    tile_col = tid % BLOCKSIZE

    # 3) Global indices (x = row in C, y = col in C)
    #    run_k2 uses grid = (ceil(M/32), ceil(N/32)), so:
    #    blockIdx.x selects the row-tile, blockIdx.y selects the col-tile.
    x = cuda.blockIdx.x * BLOCKSIZE + tile_row
    y = cuda.blockIdx.y * BLOCKSIZE + tile_col

    # 4) Bounds check for edge tiles
    if x < M and y < N:
        # 5) Accumulator (float32) for the dot product
        tmp = float32(0.0)

        # 6) Same math as K1: dot(A[x,:], B[:,y])
        for i in range(K):
            tmp += A[x, i] * B[i, y]

        # 7) Store result
        C[x, y] = tmp

    return


# ── K3: shared-memory cache-blocking (TODO) ─────────────────────────

@cuda.jit
def sgemm_smem(A, B, C, M, N, K):
    """K3: stream the K dimension in chunks of BK3. Each block computes a
            BM3 x BN3 output tile by repeatedly:
        1. cooperatively loading a BM3 x BK3 slice of A and a BK3 x BN3
           slice of B into shared memory (one element per thread per slice),
        2. cuda.syncthreads(),
        3. dotting the row of As into the column of Bs to update one
           per-thread accumulator,
        4. cuda.syncthreads() before the next K-chunk.

    Launch shape (run_k3 below uses this):
        block = (BM3 * BN3,)                    # 1024 threads, 1D
        grid  = (ceil(M / BM3), ceil(N / BN3))

    Use cuda.shared.array((BM3, BK3), float32) for As and a similar
    (BK3, BN3) for Bs.
    Use 0.0 in the SMEM load when the global index is out of bounds.
    """
    # --- Thread mapping: 1D block of 1024 threads -> 2D (row, col) in 32x32 tile ---
    tid = cuda.threadIdx.x                       # 0..1023
    row = tid // BN3                             # 0..31
    col = tid - row * BN3                        # 0..31  (slightly faster than %)

    # --- Block origin in C ---
    x = cuda.blockIdx.x * BM3 + row              # global row index in C
    y = cuda.blockIdx.y * BN3 + col              # global col index in C

    # --- Shared memory tiles (compile-time constant shapes) ---
    As = cuda.shared.array((BM3, BK3), float32)  # 32x32 tile from A
    Bs = cuda.shared.array((BK3, BN3), float32)  # 32x32 tile from B

    # --- Accumulator in registers ---
    acc = float32(0.0)

    # --- Loop over K dimension in tiles of size BK3 ---
    # k0 is the starting K index for the current tile
    for k0 in range(0, K, BK3):

        # Global indices for the elements this thread will load into shared memory
        a_col = k0 + col                         # column in A tile (K dimension)
        b_row = k0 + row                         # row in B tile (K dimension)

        # Load A tile element (coalesced across col for a fixed row)
        if x < M and a_col < K:
            As[row, col] = A[x, a_col]
        else:
            As[row, col] = float32(0.0)

        # Load B tile element (coalesced across col for a fixed b_row)
        if b_row < K and y < N:
            Bs[row, col] = B[b_row, y]
        else:
            Bs[row, col] = float32(0.0)

        # Ensure the full tiles are loaded before using them
        cuda.syncthreads()

        # Compute partial dot product for this (x, y) using the shared tiles
        # acc += sum_{kk=0..BK3-1} As[row, kk] * Bs[kk, col]
        for kk in range(BK3):
            acc += As[row, kk] * Bs[kk, col]

        # Ensure all threads are done reading shared memory before it gets overwritten
        cuda.syncthreads()

    # Write the result
    if x < M and y < N:
        C[x, y] = acc

    return  

# ── K4: 1D register tiling (TODO) ───────────────────────────────────

@cuda.jit
def sgemm_1d_tile(A, B, C, M, N, K):
    """K4: extend K3 by giving each thread TM4 = 8 rows in a single column
    of the BM4 x BN4 output tile.

    Note: blockIdx.x now indexes COLUMNS of the output.
    The run_k4 wrapper below already accounts for this, but you need to compute the global (row, col)
    start of your block accordingly.

    Launch shape (run_k4 below uses this):
        block = ((BM4 * BN4) // TM4,)           # 512 threads
        grid  = (ceil(N / BN4), ceil(M / BM4))  # x = col, y = row

    Cooperative loads here are tidy: A's tile is BM4 x BK4 = 512 elements,
    B's tile is BK4 x BN4 = 512 elements, and you have 512 threads so
    exactly one element per thread per tile (so no inner-load loop)

    Use cuda.local.array(TM4, float32) for the per-thread accumulator array.
    Initialize all entries to 0.0 before the K-loop.
    """
    tid = cuda.threadIdx.x  # 0..511 because run_k4 uses 512 threads/block [3](https://ucsc0-my.sharepoint.com/personal/rjulianc_ucsc_edu/Documents/Microsoft%20Copilot%20Chat%20Files/kernels.py)

    # ----------------------------
    # (A) Map thread -> output micro-tile
    # ----------------------------
    # There are BN4=64 columns. For each column, we need BM4/TM4=8 threads
    # stacked in the row direction to cover 64 rows with 8 rows/thread.
    row_group = tid // BN4                 # 0..7
    col = tid - row_group * BN4            # 0..63  (faster than %)

    # run_k4 wrapper uses axis swap: blockIdx.x indexes columns (N), blockIdx.y indexes rows (M). [3](https://ucsc0-my.sharepoint.com/personal/rjulianc_ucsc_edu/Documents/Microsoft%20Copilot%20Chat%20Files/kernels.py)
    block_col0 = cuda.blockIdx.x * BN4     # starting column of this 64-wide tile
    block_row0 = cuda.blockIdx.y * BM4     # starting row of this 64-tall tile

    y = block_col0 + col                   # global column index for this thread's outputs
    row_base = block_row0 + row_group * TM4  # global row index of the first of this thread's 8 outputs

    # ----------------------------
    # (B) Shared memory tiles
    # ----------------------------
    # As: 64x8 = 512 elements  -> perfect: one per thread
    # Bs: 8x64 = 512 elements  -> perfect: one per thread
    As = cuda.shared.array((BM4, BK4), float32)
    Bs = cuda.shared.array((BK4, BN4), float32)

    # ----------------------------
    # (C) Register accumulators (aggressive: scalars, not arrays)
    # ----------------------------
    acc0 = float32(0.0)
    acc1 = float32(0.0)
    acc2 = float32(0.0)
    acc3 = float32(0.0)
    acc4 = float32(0.0)
    acc5 = float32(0.0)
    acc6 = float32(0.0)
    acc7 = float32(0.0)

    # ----------------------------
    # (D) Stream K in tiles of BK4=8
    # ----------------------------
    for k0 in range(0, K, BK4):

        # ----------------------------
        # (D1) Cooperative load A tile (64x8) into shared memory
        # ----------------------------
        # Map tid (0..511) onto As[a_r, a_c] with a_r in 0..63, a_c in 0..7
        a_r = tid >> 3              # tid // 8
        a_c = tid - (a_r << 3)      # tid % 8 (faster than %)
        ax = block_row0 + a_r       # global row in A
        ak = k0 + a_c               # global k in A

        if ax < M and ak < K:
            As[a_r, a_c] = A[ax, ak]
        else:
            As[a_r, a_c] = float32(0.0)

        # ----------------------------
        # (D2) Cooperative load B tile (8x64) into shared memory
        # ----------------------------
        # Map tid (0..511) onto Bs[b_r, b_c] with b_r in 0..7, b_c in 0..63
        b_r = tid >> 6              # tid // 64
        b_c = tid - (b_r << 6)      # tid % 64
        bk = k0 + b_r               # global k in B
        by = block_col0 + b_c       # global col in B

        if bk < K and by < N:
            Bs[b_r, b_c] = B[bk, by]
        else:
            Bs[b_r, b_c] = float32(0.0)

        cuda.syncthreads()

        # ----------------------------
        # (D3) Compute: 8 outputs/thread using shared tiles
        # ----------------------------
        # Each thread uses column 'col' of Bs and rows row_group*8..row_group*8+7 of As.
        # Unroll the "r" dimension manually for register efficiency.
        r0 = row_group * TM4

        # inner-k loop is small and constant (BK4=8), good for unrolling by compiler
        for kk in range(BK4):
            b = Bs[kk, col]
            acc0 += As[r0 + 0, kk] * b
            acc1 += As[r0 + 1, kk] * b
            acc2 += As[r0 + 2, kk] * b
            acc3 += As[r0 + 3, kk] * b
            acc4 += As[r0 + 4, kk] * b
            acc5 += As[r0 + 5, kk] * b
            acc6 += As[r0 + 6, kk] * b
            acc7 += As[r0 + 7, kk] * b

        cuda.syncthreads()

    # ----------------------------
    # (E) Write results (guard bounds)
    # ----------------------------
    if y < N:
        x0 = row_base
        if x0 + 0 < M: C[x0 + 0, y] = acc0
        if x0 + 1 < M: C[x0 + 1, y] = acc1
        if x0 + 2 < M: C[x0 + 2, y] = acc2
        if x0 + 3 < M: C[x0 + 3, y] = acc3
        if x0 + 4 < M: C[x0 + 4, y] = acc4
        if x0 + 5 < M: C[x0 + 5, y] = acc5
        if x0 + 6 < M: C[x0 + 6, y] = acc6
        if x0 + 7 < M: C[x0 + 7, y] = acc7
    return

# ── K5: 2D register tiling (TODO) ───────────────────────────────────

@cuda.jit
def sgemm_2d_tile(A, B, C, M, N, K):
    """K5: extend K4 to a TM5 x TN5 = 8 x 8 register tile per thread.
    Inside the inner-k loop, cache TM5 As values and TN5 Bs values into
    register arrays, then do the TM5 x TN5 outer-product update.

    Launch shape (run_k5 below uses this):
        block = ((BM5 * BN5) // (TM5 * TN5),)   # 256 threads
        grid  = (ceil(N / BN5), ceil(M / BM5))

    Cooperative loads now need a stride loop: the tile has more elements
    (BM5 * BK5 = 1024) than the block has threads (256), so each thread
    loads BM5 * BK5 / 256 = 4 elements of A per K-chunk and similarly for B.
    Pick the per-thread row stride so that consecutive threads touch
    consecutive memory addresses (= coalesced GMEM loads).

    For accumulators, use cuda.local.array((TM5, TN5), float32).
    Numba supports tuple-shaped local arrays!
    """
    tid = cuda.threadIdx.x  # 0..255 (wrapper launches 256 threads) [3](https://ucsc0-my.sharepoint.com/personal/rjulianc_ucsc_edu/Documents/Microsoft%20Copilot%20Chat%20Files/kernels.py)

    # ----------------------------
    # (A) Thread -> micro-tile mapping
    # ----------------------------
    # There are (BM5/TM5)=16 micro-tiles in rows and (BN5/TN5)=16 in cols.
    # Total micro-tiles = 16*16 = 256 threads.
    micro_cols = BN5 // TN5  # 128/8 = 16
    micro_row = tid // micro_cols       # 0..15
    micro_col = tid - micro_row * micro_cols  # 0..15

    # Axis swap in wrapper: blockIdx.x indexes columns (N), blockIdx.y indexes rows (M). [3](https://ucsc0-my.sharepoint.com/personal/rjulianc_ucsc_edu/Documents/Microsoft%20Copilot%20Chat%20Files/kernels.py)
    block_col0 = cuda.blockIdx.x * BN5
    block_row0 = cuda.blockIdx.y * BM5

    row_base = block_row0 + micro_row * TM5   # top row of this thread's 8x8 tile
    col_base = block_col0 + micro_col * TN5   # left col of this thread's 8x8 tile

    # ----------------------------
    # (B) Shared memory tiles
    # ----------------------------
    # As: 128x8 (1024 elements), Bs: 8x128 (1024 elements) [3](https://ucsc0-my.sharepoint.com/personal/rjulianc_ucsc_edu/Documents/Microsoft%20Copilot%20Chat%20Files/kernels.py)
    As = cuda.shared.array((BM5, BK5), float32)
    Bs = cuda.shared.array((BK5, BN5), float32)

    # ----------------------------
    # (C) Register accumulators: 8x8
    # ----------------------------
    acc = cuda.local.array((TM5, TN5), float32)
    for i in range(TM5):
        for j in range(TN5):
            acc[i, j] = float32(0.0)

    # Temporary register caches for one kk slice
    a_reg = cuda.local.array((TM5,), float32)
    b_reg = cuda.local.array((TN5,), float32)

    # ----------------------------
    # (D) Stream K in tiles of BK5=8
    # ----------------------------
    for k0 in range(0, K, BK5):

        # ----------------------------
        # (D1) Cooperative loads into shared memory
        # ----------------------------
        # Total elements to load:
        #   As: 128*8 = 1024
        #   Bs: 8*128 = 1024
        # Threads/block = 256, so each thread loads 4 elements of As and 4 of Bs.
        for t in range(4):
            # Load one element of As
            idxA = tid + t * 256                # 0..1023
            a_r = idxA >> 3                     # idxA // 8  (0..127)
            a_c = idxA - (a_r << 3)             # idxA % 8   (0..7)
            ax = block_row0 + a_r               # global row in A
            ak = k0 + a_c                       # global k in A
            if ax < M and ak < K:
                As[a_r, a_c] = A[ax, ak]
            else:
                As[a_r, a_c] = float32(0.0)

            # Load one element of Bs
            idxB = tid + t * 256                # 0..1023
            b_r = idxB >> 7                     # idxB // 128 (0..7)
            b_c = idxB - (b_r << 7)             # idxB % 128  (0..127)
            bk = k0 + b_r                       # global k in B
            by = block_col0 + b_c               # global col in B
            if bk < K and by < N:
                Bs[b_r, b_c] = B[bk, by]
            else:
                Bs[b_r, b_c] = float32(0.0)

        cuda.syncthreads()

        # ----------------------------
        # (D2) Compute: outer-product updates for kk=0..7
        # ----------------------------
        for kk in range(BK5):
            # Cache 8 A values for this kk into registers
            a_reg[0] = As[micro_row * TM5 + 0, kk]
            a_reg[1] = As[micro_row * TM5 + 1, kk]
            a_reg[2] = As[micro_row * TM5 + 2, kk]
            a_reg[3] = As[micro_row * TM5 + 3, kk]
            a_reg[4] = As[micro_row * TM5 + 4, kk]
            a_reg[5] = As[micro_row * TM5 + 5, kk]
            a_reg[6] = As[micro_row * TM5 + 6, kk]
            a_reg[7] = As[micro_row * TM5 + 7, kk]

            # Cache 8 B values for this kk into registers
            b_reg[0] = Bs[kk, micro_col * TN5 + 0]
            b_reg[1] = Bs[kk, micro_col * TN5 + 1]
            b_reg[2] = Bs[kk, micro_col * TN5 + 2]
            b_reg[3] = Bs[kk, micro_col * TN5 + 3]
            b_reg[4] = Bs[kk, micro_col * TN5 + 4]
            b_reg[5] = Bs[kk, micro_col * TN5 + 5]
            b_reg[6] = Bs[kk, micro_col * TN5 + 6]
            b_reg[7] = Bs[kk, micro_col * TN5 + 7]

            # Outer product: acc[i,j] += a_reg[i] * b_reg[j]
            for i in range(TM5):
                ai = a_reg[i]
                acc[i, 0] += ai * b_reg[0]
                acc[i, 1] += ai * b_reg[1]
                acc[i, 2] += ai * b_reg[2]
                acc[i, 3] += ai * b_reg[3]
                acc[i, 4] += ai * b_reg[4]
                acc[i, 5] += ai * b_reg[5]
                acc[i, 6] += ai * b_reg[6]
                acc[i, 7] += ai * b_reg[7]

        cuda.syncthreads()

    # ----------------------------
    # (E) Write back the 8x8 micro-tile to C (bounds-guarded)
    # ----------------------------
    for i in range(TM5):
        x = row_base + i
        if x < M:
            y0 = col_base
            if y0 + 0 < N: C[x, y0 + 0] = acc[i, 0]
            if y0 + 1 < N: C[x, y0 + 1] = acc[i, 1]
            if y0 + 2 < N: C[x, y0 + 2] = acc[i, 2]
            if y0 + 3 < N: C[x, y0 + 3] = acc[i, 3]
            if y0 + 4 < N: C[x, y0 + 4] = acc[i, 4]
            if y0 + 5 < N: C[x, y0 + 5] = acc[i, 5]
            if y0 + 6 < N: C[x, y0 + 6] = acc[i, 6]
            if y0 + 7 < N: C[x, y0 + 7] = acc[i, 7]
    return


# ── Launch wrappers (provided — do not edit) ────────────────────────

def run_k1(A, B, C, M, N, K):
    grid = (math.ceil(M / BLOCKSIZE), math.ceil(N / BLOCKSIZE))
    block = (BLOCKSIZE, BLOCKSIZE)
    sgemm_naive[grid, block](A, B, C, M, N, K)


def run_k2(A, B, C, M, N, K):
    grid = (math.ceil(M / BLOCKSIZE), math.ceil(N / BLOCKSIZE))
    block = (BLOCKSIZE * BLOCKSIZE,)
    sgemm_coalesced[grid, block](A, B, C, M, N, K)


def run_k3(A, B, C, M, N, K):
    grid = (math.ceil(M / BM3), math.ceil(N / BN3))
    block = (BM3 * BN3,)
    sgemm_smem[grid, block](A, B, C, M, N, K)


def run_k4(A, B, C, M, N, K):
    # Axis swap: blockIdx.x indexes columns of C.
    grid = (math.ceil(N / BN4), math.ceil(M / BM4))
    block = ((BM4 * BN4) // TM4,)
    sgemm_1d_tile[grid, block](A, B, C, M, N, K)


def run_k5(A, B, C, M, N, K):
    grid = (math.ceil(N / BN5), math.ceil(M / BM5))
    block = ((BM5 * BN5) // (TM5 * TN5),)
    sgemm_2d_tile[grid, block](A, B, C, M, N, K)


# Graded kernels in the order the rubric uses (1/4 → C, 2/4 → B-, ...).
KERNELS = [
    ("k2_coalesce", run_k2),
    ("k3_smem",     run_k3),
    ("k4_1d_tile",  run_k4),
    ("k5_2d_tile",  run_k5),
]

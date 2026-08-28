# Hybrid: the rc kernel on the "nice" rows (heavy rows emptied in place, so
# the kernel writes α·0 + β·C for them) plus the "odd" (heavy) rows as a
# small compressed CSR (nodd × k): one simple KA kernel sums interleaved
# slices of each odd row (chunk c takes entries c, c+nchunks, c+2·nchunks, …,
# so neighbouring threads read neighbouring entries) into a padded
# (nchunks × nodd × n) slab of partials, which a tree-based `sum` collapses
# before adding into C — deterministic, ~log-depth rounding. Chunks are
# short (ODD_CHUNK entries) so the GPU is busy: 92k entries / 32 = 2.9k
# chunks per row and column. The vision instances
# have two rows with ~92k entries (max-flow source/sink) vs ≤ 14 elsewhere,
# which collapse a row-per-lane kernel.

# KernelAbstractions is an indirect dependency; load it by UUID
if !@isdefined(KA)
    const KA = Base.require(Base.PkgId(
        Base.UUID("63c18a36-062a-441e-b654-da1e3ab1ce7c"), "KernelAbstractions"))
end
using .KA

struct HybridSparseMatrix{T, Ti, M <: SplitRangeCSRMatrix{T, Ti}, V, Vi} <: AbstractMatrix{T}
    nice_part::M
    odd_rows::Vi                                    # row index in C of each odd row
    odd_rowptr::Vi; odd_colval::Vi; odd_nzval::V    # compressed CSR (nodd × k) of the odd rows
    odd_maxchunks::Int                              # chunks of the longest odd row
end
Base.size(A::HybridSparseMatrix) = size(A.nice_part)
const ODD_CHUNK = 32                                # entries per chunk (thread)

"Move a compressed CSR of the odd rows (host arrays) to the device of `nice`."
function HybridSparseMatrix(nice::SplitRangeCSRMatrix{T, Ti}, rows, rowptr, colval, nzval) where {T, Ti}
    maxchunks = maximum(cld(rowptr[h + 1] - rowptr[h], ODD_CHUNK) for h in eachindex(rows); init=0)
    dev(x) = adapt(typeof(nice.contiguous_lo).name.wrapper, x)   # CuArray etc.
    HybridSparseMatrix(nice, dev(Ti.(rows)), dev(Ti.(rowptr)), dev(Ti.(colval)), dev(T.(nzval)), maxchunks)
end

# use wide enough type in TileArray construction, otherwise we get crashes on large instances
function launch_rc!(C, A::SplitRangeCSRMatrix{T, Ti}, B, α, β, tm, tn, tk, beta_nz) where {T, Ti}
    grid = (cld(size(C, 1), tm), cld(size(C, 2), tn))
    wrap(X) = ct.TileArray(X; index=length(X) > typemax(Int32) ? Int64 : Ti)
    ct.launch(spmm_rc_kernel, grid, wrap(C), A, wrap(B), α, β,
              ct.Constant(tm), ct.Constant(tn), ct.Constant(tk), ct.Constant(beta_nz))
end

"part[c, h, j] = Σ of nzval · B[colval, j] over entries c-1, c-1+nchunks, … of odd row h"
@kernel function odd_partials!(part, rowptr, colval, nzval, B, nchunks)
    c, h, j = @index(Global, NTuple)
    @inbounds begin
        s = zero(eltype(part))
        for p in (rowptr[h] + c - 1):nchunks:(rowptr[h + 1] - 1)   # empty for padding chunks
            s = muladd(nzval[p], B[colval[p], j], s)
        end
        part[c, h, j] = s
    end
end

function spmm_hybrid!(C, A::HybridSparseMatrix, B, α, β, tm, tn, tk, beta_nz;
                      part=similar(C, A.odd_maxchunks, length(A.odd_rows), size(C, 2)))
    launch_rc!(C, A.nice_part, B, α, β, tm, tn, tk, beta_nz)
    odd_rows!(C, A, B, α; part)
    return C
end

"C[rows[h], j] += α · sums[1, h, j]"
@kernel function odd_add!(C, rows, sums, α)
    h, j = @index(Global, NTuple)
    @inbounds C[rows[h], j] += α * sums[1, h, j]
end

"Odd rows only: chunk partials, tree reduction, add into C (which holds β·C there).
Everything stays on the device: `view(C, rows, :) .+= …` with a device index
vector runs `checkbounds` as a GPU reduction copied back to the host, and
that synchronization was worth 0.1–15 ms of idle GPU per call."
function odd_rows!(C, A::HybridSparseMatrix{T}, B, α; part) where {T}
    odd_partials!(C, A, B; part)
    sums = sum(part; dims=1)
    odd_add!(KA.get_backend(C), 256)(C, A.odd_rows, sums, T(α); ndrange=size(sums)[2:3])
    return C
end

odd_partials!(C, A::HybridSparseMatrix, B; part) =
    odd_partials!(KA.get_backend(C), 256)(part, A.odd_rowptr, A.odd_colval, A.odd_nzval, B,
                                          A.odd_maxchunks; ndrange=size(part))

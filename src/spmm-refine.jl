using Pkg
Pkg.add("StatsBase")
Pkg.add("JSON")
Pkg.add("BenchmarkTools")
Pkg.add("SparseArrays")
Pkg.add("SuiteSparseGraphBLAS")
Pkg.add("Finch")
Pkg.add("JSON3")

using StatsBase
using JSON
using BenchmarkTools
using SparseArrays
using SuiteSparseGraphBLAS
using Finch
using JSON3


const SIZE = 4096


# s / SIZE is the density of the matrix
function create_sparse_mat(s, M = SIZE)
    # initialize u with random int
    u = zeros(M, M)

    # fill in s entries
    all_coords = collect(Iterators.product(1:M, 1:M))
    coords = sample(all_coords, s, replace=false)
    
    for coord in coords
        i, j = coord
        
        u[i, j] = rand(Float64)
    end

    return u
end


function custom_coo_setup(M)
    u = []

    for i in 1:SIZE
        for j in 1:SIZE
            if M[i, j] != 0
                push!(u, (i, j, M[i, j]))
            end
        end
    end

    return u
end

function custom_coo_mul(M, X)
    y = zeros(SIZE, SIZE)

    for (i, j, v) in M
        for d in 1:SIZE
            y[i, d] += v * X[j, d]
        end
    end

    return y
end


function isstruct(x)
    return x isa Any && !(x isa Number) && !(x isa AbstractArray)
end


function finch_isapprox(a, b; rtol=1e-8, atol=1e-12)
    if typeof(a) != typeof(b)
        # println("type diff: ", typeof(a), " vs ", typeof(b))
        return false
    end

    if a isa Float64 && b isa Float64
        return abs(a - b) ≤ atol + rtol * max(abs(a), abs(b))
    elseif a isa Integer && b isa Integer
        return a == b
    elseif a isa AbstractArray && b isa AbstractArray
        if size(a) != size(b)
            # println("size diff: ", size(a), " vs ", size(b))
            return false
        end

        for (x, y) in zip(a, b)
            if !finch_isapprox(x, y; rtol=rtol, atol=atol)
                # println("element diff: ", x, " vs ", y)
                return false
            end
        end
        return true

    elseif a isa Finch.Tensor && b isa Finch.Tensor
        if size(a) != size(b)
            # println("tensor size diff: ", size(a), " vs ", size(b))
            return false
        end
        for I in CartesianIndices(size(a))
            av = a[I]
            bv = b[I]
            if !(abs(av - bv) ≤ atol + rtol * max(abs(av), abs(bv)))
                # println("tensor value diff at $I: $av vs $bv")
                return false
            end
        end
        return true

    elseif isstruct(a) && isstruct(b)
        for name in fieldnames(typeof(a))
            ax = getfield(a, name)
            bx = getfield(b, name)
            if !finch_isapprox(ax, bx; rtol=rtol, atol=atol)
                # println("field ", name, " diff: ", ax, " vs ", bx)
                return false
            end
        end
        return true
    else
        return isequal(a, b)
    end
end


function sa_csc(M, X, sol)
    m_sparse = sparse(M)

    bench_results = @benchmark $m_sparse * $X
    temp = m_sparse * X
    if isapprox(temp, sol; rtol=1e-8, atol=1e-12)
        return (minimum(bench_results.times))
    else
        return -1
    end
end


function ssgblas_mt(M, X, sol)
    m_SSGBLAS = GBMatrix(M)
    x_SSGBLAS = GBMatrix(X)

    bench_results = @benchmark $m_SSGBLAS * $x_SSGBLAS
    temp = m_SSGBLAS * x_SSGBLAS
    if isapprox(temp, sol; rtol=1e-8, atol=1e-12)
        return (minimum(bench_results.times))
    else
        return -1
    end
end


function custom_coo(M, X, sol)
    m_coo = custom_coo_setup(M)

    bench_results = @benchmark custom_coo_mul($m_coo, $X)
    temp = custom_coo_mul(m_coo, X)
    if isapprox(temp, sol; rtol=1e-8, atol=1e-12)
        return (minimum(bench_results.times))
    else
        return -1
    end
end


function fin_csc(M, X, sol)
    m_ten = Finch.Tensor(CSCFormat(), M)
    x_ten = Finch.Tensor(CSCFormat(), X)
    sol_ten = Finch.Tensor(CSCFormat(), sol)

    bench_results = @benchmark $m_ten * $x_ten
    temp = m_ten * x_ten
    temp_ten = Finch.Tensor(CSCFormat(), temp)
    if finch_isapprox(temp_ten, sol_ten; rtol=1e-8, atol=1e-12)
        return (minimum(bench_results.times))
    else
        return -1
    end
end

function fin_csf(M, X, sol)
    m_ten = Finch.Tensor(CSFFormat(2), M)
    x_ten = Finch.Tensor(CSFFormat(2), X)
    sol_ten = Finch.Tensor(CSFFormat(2), sol)

    bench_results = @benchmark $m_ten * $x_ten
    temp = m_ten * x_ten
    temp_ten = Finch.Tensor(CSFFormat(2), temp)
    if finch_isapprox(temp_ten, sol_ten; rtol=1e-8, atol=1e-12)
        return (minimum(bench_results.times))
    else
        return -1
    end
end

function fin_dcsc(M, X, sol)
    m_ten = Finch.Tensor(DCSCFormat(), M)
    x_ten = Finch.Tensor(DCSCFormat(), X)
    sol_ten = Finch.Tensor(DCSCFormat(), sol)

    bench_results = @benchmark $m_ten * $x_ten
    temp = m_ten * x_ten
    temp_ten = Finch.Tensor(DCSCFormat(), temp)
    if finch_isapprox(temp_ten, sol_ten; rtol=1e-8, atol=1e-12)
        return (minimum(bench_results.times))
    else
        return -1
    end
end

function fin_dcsf(M, X, sol)
    m_ten = Finch.Tensor(DCSFFormat(2), M)
    x_ten = Finch.Tensor(DCSFFormat(2), X)
    sol_ten = Finch.Tensor(DCSFFormat(2), sol)

    bench_results = @benchmark $m_ten * $x_ten
    temp = m_ten * x_ten
    temp_ten = Finch.Tensor(DCSFFormat(2), temp)
    if finch_isapprox(temp_ten, sol_ten; rtol=1e-8, atol=1e-12)
        return (minimum(bench_results.times))
    else
        return -1
    end
end

function fin_coo(M, X, sol)
    m_ten = Finch.Tensor(COOFormat(2), M)
    x_ten = Finch.Tensor(COOFormat(2), X)
    sol_ten = Finch.Tensor(COOFormat(2), sol)

    bench_results = @benchmark $m_ten * $x_ten
    temp = m_ten * x_ten
    temp_ten = Finch.Tensor(COOFormat(2), temp)
    if finch_isapprox(temp_ten, sol_ten; rtol=1e-8, atol=1e-12)
        return (minimum(bench_results.times))
    else
        return -1
    end
end

function fin_hash(M, X, sol)
    m_ten = Finch.Tensor(HashFormat(2), M)
    x_ten = Finch.Tensor(HashFormat(2), X)
    sol_ten = Finch.Tensor(HashFormat(2), sol)

    bench_results = @benchmark $m_ten * $x_ten
    temp = m_ten * x_ten
    temp_ten = Finch.Tensor(HashFormat(2), temp)
    if finch_isapprox(temp_ten, sol_ten; rtol=1e-8, atol=1e-12)
        return (minimum(bench_results.times))
    else
        return -1
    end
end

function fin_bytemap(M, X, sol)
    m_ten = Finch.Tensor(ByteMapFormat(2), M)
    x_ten = Finch.Tensor(ByteMapFormat(2), X)
    sol_ten = Finch.Tensor(ByteMapFormat(2), sol)

    bench_results = @benchmark $m_ten * $x_ten
    temp = m_ten * x_ten
    temp_ten = Finch.Tensor(ByteMapFormat(2), temp)
    if finch_isapprox(temp_ten, sol_ten; rtol=1e-8, atol=1e-12)
        return (minimum(bench_results.times))
    else
        return -1
    end
end


inx = 0
inc = 1
cnt = 0
pairs_sa_csc = []
pairs_ssgblas = []
pairs_coo = []
pairs_fin_csc = []
pairs_fin_csf = []
pairs_fin_dcsc = []
pairs_fin_dcsf = []
pairs_fin_coo = []
pairs_fin_hash = []
pairs_fin_bytemap = []

X = JSON3.read("ben/x.json", Vector{Vector{Float64}})
X = reduce(hcat, X) |> permutedims


while inx <= (SIZE / 2)
    M_dense = JSON3.read(string("ben/m", inx, ".json"), Vector{Vector{Float64}})
    M_dense = reduce(hcat, M_dense) |> permutedims

    sol = JSON3.read(string("ben/sol", inx,".json"), Vector{Vector{Float64}})
    sol = reduce(hcat, sol) |> permutedims

    println("Density: ", inx / SIZE)

    local sa_csc_val = sa_csc(M_dense, X, sol)
    if sa_csc_val != -1
        println("\tSparseArrays CSC: \t", sa_csc_val, "ns")
        push!(pairs_sa_csc, (inx / SIZE, sa_csc_val))
    else
        println("\tSparseArray CSC: matmul resulted in wrong answer.")
    end


    local ssgblas = ssgblas_mt(M_dense, X, sol)
    if ssgblas != -1
        println("\tSSGBLAS: \t\t", ssgblas, "ns")
        push!(pairs_ssgblas, (inx / SIZE, ssgblas))
    else
        println("\tSSGBLAS: matmul resulted in wrong answer.")
    end


    local custom_coo_val = custom_coo(M_dense, X, sol)
    if custom_coo_val != -1
        println("\tcustom COO: \t\t", custom_coo_val, "ns")
        push!(pairs_coo, (inx / SIZE, custom_coo_val))
    else
        println("\tcustom COO: matmul resulted in wrong answer.")
    end


    local fin_csc_val = fin_csc(M_dense, X, sol)
    if fin_csc_val != -1
        println("\tfinch CSC: \t\t", fin_csc_val, "ns")
        push!(pairs_fin_csc, (inx / SIZE, fin_csc_val))
    else
        println("\tfinch CSC: matmul resulted in wrong answer.")
    end


    local fin_csf_val = fin_csf(M_dense, X, sol)
    if fin_csf_val != -1
        println("\tfinch CSF: \t\t", fin_csf_val, "ns")
        push!(pairs_fin_csf, (inx / SIZE, fin_csf_val))
    else
        println("\tfinch CSF: matmul resulted in wrong answer.")
    end


    local fin_dcsc_val = fin_dcsc(M_dense, X, sol)
    if fin_dcsc_val != -1
        println("\tfinch DCSC: \t\t", fin_dcsc_val, "ns")
        push!(pairs_fin_dcsc, (inx / SIZE, fin_dcsc_val))
    else
        println("\tfinch DCSC: matmul resulted in wrong answer.")
    end


    local fin_dcsf_val = fin_dcsf(M_dense, X, sol)
    if fin_dcsf_val != -1
        println("\tfinch DCSF: \t\t", fin_dcsf_val, "ns")
        push!(pairs_fin_dcsf, (inx / SIZE, fin_dcsf_val))
    else
        println("\tfinch DCSF: matmul resulted in wrong answer.")
    end


    local fin_coo_val = fin_coo(M_dense, X, sol)
    if fin_coo_val != -1
        println("\tfinch COO: \t\t", fin_coo_val, "ns")
        push!(pairs_fin_coo, (inx / SIZE, fin_coo_val))
    else
        println("\tfinch COO: matmul resulted in wrong answer.")
    end


    local fin_hash_val = fin_hash(M_dense, X, sol)
    if fin_hash_val != -1
        println("\tfinch Hash: \t\t", fin_hash_val, "ns")
        push!(pairs_fin_hash, (inx / SIZE, fin_hash_val))
    else
        println("\tfinch Hash: matmul resulted in wrong answer.")
    end


    local fin_bm_val = fin_bytemap(M_dense, X, sol)
    if fin_bm_val != -1
        println("\tfinch Bytemap: \t\t", fin_bm_val, "ns")
        push!(pairs_fin_bytemap, (inx / SIZE, fin_bm_val))
    else
        println("\tfinch Bytemap: matmul resulted in wrong answer.")
    end

    if cnt >= 16
        global cnt = 0
        global inc *= 2
    end

    global inx += inc
    global cnt += 1
end

open("res/sparse-arrays-csc.json", "w") do f
    write(f, JSON.json(pairs_sa_csc))
end

open("res/suite-sparse-graph-blas.json", "w") do f
    write(f, JSON.json(pairs_ssgblas))
end

open("res/custom-coo.json", "w") do f
    write(f, JSON.json(pairs_coo))
end

open("res/finch-csc.json", "w") do f
    write(f, JSON.json(pairs_fin_csc))
end

open("res/finch-csf.json", "w") do f
    write(f, JSON.json(pairs_fin_csf))
end

open("res/finch-dcsc.json", "w") do f
    write(f, JSON.json(pairs_fin_dcsc))
end

open("res/finch-dcsf.json", "w") do f
    write(f, JSON.json(pairs_fin_dcsf))
end

open("res/finch-coo.json", "w") do f
    write(f, JSON.json(pairs_fin_coo))
end

open("res/finch-hash.json", "w") do f
    write(f, JSON.json(pairs_fin_hash))
end

open("res/finch-bm.json", "w") do f
    write(f, JSON.json(pairs_fin_bytemap))
end
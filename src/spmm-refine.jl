using Pkg
# Pkg.add("StatsBase")
# Pkg.add("JSON")
# Pkg.add("BenchmarkTools")
# Pkg.add("SparseArrays")
# Pkg.add("SuiteSparseGraphBLAS")
# Pkg.add("Finch")

using StatsBase
# using JSON
using BenchmarkTools
using SparseArrays
using SuiteSparseGraphBLAS
using Finch


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


function sa_csc(M, X, sol)
    m_sparse = sparse(M)

    bench_results = @benchmark $m_sparse * $X
    temp = m_sparse * X
    if isequal(temp, sol)
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
    if isequal(temp, sol)
        return (minimum(bench_results.times))
    else
        return -1
    end
end


function custom_coo(M, X, sol)
    m_coo = custom_coo_setup(M)

    bench_results = @benchmark custom_coo_mul($m_coo, $X)
    temp = custom_coo_mul(m_coo, X)
    if isequal(temp, sol)
        return (minimum(bench_results.times))
    else
        return -1
    end
end


function fin_csc(M, X, sol)
    m_ten = Finch.Tensor(CSCFormat(), M)
    x_ten = Finch.Tensor(CSCFormat(), X)

    bench_results = @benchmark $m_ten * $x_ten
    temp = m_ten * x_ten
    if isequal(temp, sol)
        return (minimum(bench_results.times))
    else
        return -1
    end
end

function fin_csf(M, X, sol)
    m_ten = Finch.Tensor(CSFFormat(2), M)
    x_ten = Finch.Tensor(CSFFormat(2), X)

    bench_results = @benchmark $m_ten * $x_ten
    temp = m_ten * x_ten
    if isequal(temp, sol)
        return (minimum(bench_results.times))
    else
        return -1
    end
end

function fin_dcsc(M, X, sol)
    m_ten = Finch.Tensor(DCSCFormat(), M)
    x_ten = Finch.Tensor(DCSCFormat(), X)

    bench_results = @benchmark $m_ten * $x_ten
    temp = m_ten * x_ten
    if isequal(temp, sol)
        return (minimum(bench_results.times))
    else
        return -1
    end
end

function fin_dcsf(M, X, sol)
    m_ten = Finch.Tensor(DCSFFormat(2), M)
    x_ten = Finch.Tensor(DCSFFormat(2), X)

    bench_results = @benchmark $m_ten * $x_ten
    temp = m_ten * x_ten
    if isequal(temp, sol)
        return (minimum(bench_results.times))
    else
        return -1
    end
end

function fin_coo(M, X, sol)
    m_ten = Finch.Tensor(COOFormat(2), M)
    x_ten = Finch.Tensor(COOFormat(2), X)

    bench_results = @benchmark $m_ten * $x_ten
    temp = m_ten * x_ten
    if isequal(temp, sol)
        return (minimum(bench_results.times))
    else
        return -1
    end
end

function fin_hash(M, X, sol)
    m_ten = Finch.Tensor(HashFormat(2), M)
    x_ten = Finch.Tensor(HashFormat(2), X)

    bench_results = @benchmark $m_ten * $x_ten
    temp = m_ten * x_ten
    if isequal(temp, sol)
        return (minimum(bench_results.times))
    else
        return -1
    end
end

function fin_bytemap(M, X, sol)
    m_ten = Finch.Tensor(ByteMapFormat(2), M)
    x_ten = Finch.Tensor(ByteMapFormat(2), X)

    bench_results = @benchmark $m_ten * $x_ten
    temp = m_ten * x_ten
    if isequal(temp, sol)
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
while inx <= SIZE
    M_dense = create_sparse_mat(inx)
    X = randn(SIZE, SIZE)

    sol = M_dense * X

    local sa_csc_val = sa_csc(M_dense, X, sol)
    if sa_csc_val != -1
        println("Density: ", inx / SIZE, "\tCSC: ", sa_csc_val, "\tns")
        push!(pairs_sa_csc, (inx / SIZE, sa_csc_val))
    else
        println("Density: ", inx / SIZE, "\tCSC: matmul resulted in wrong answer.")
    end


    local ssgblas = ssgblas_mt(M_dense, X, sol)
    if ssgblas != -1
        println("Density: ", inx / SIZE, "\tSSGBLAS: ", ssgblas, "\tns")
        push!(pairs_ssgblas, (inx / SIZE, ssgblas))
    else
        println("Density: ", inx / SIZE, "\tSSGBLAS: matmul resulted in wrong answer.")
    end


    local custom_coo_val = custom_coo(M_dense, X, sol)
    if custom_coo_val != -1
        println("Density: ", inx / SIZE, "\tCOO: ", custom_coo_val, "\tns")
        push!(pairs_coo, (inx / SIZE, custom_coo_val))
    else
        println("Density: ", inx / SIZE, "\tCOO: matmul resulted in wrong answer.")
    end


    local fin_csc_val = fin_csc(M_dense, X, sol)
    if fin_csc_val != -1
        println("Density: ", inx / SIZE, "\tCSC: ", fin_csc_val, "\tns")
        push!(pairs_fin_csc, (inx / SIZE, fin_csc_val))
    else
        println("Density: ", inx / SIZE, "\tCSC: matmul resulted in wrong answer.")
    end


    local fin_csf_val = fin_csf(M_dense, X, sol)
    if fin_csf_val != -1
        println("Density: ", inx / SIZE, "\tCSF: ", fin_csf_val, "\tns")
        push!(pairs_fin_csf, (inx / SIZE, fin_csf_val))
    else
        println("Density: ", inx / SIZE, "\tCSF: matmul resulted in wrong answer.")
    end


    local fin_dcsc_val = fin_dcsc(M_dense, X, sol)
    if fin_dcsc_val != -1
        println("Density: ", inx / SIZE, "\tDCSC: ", fin_dcsc_val, "\tns")
        push!(pairs_fin_dcsc, (inx / SIZE, fin_dcsc_val))
    else
        println("Density: ", inx / SIZE, "\tDCSC: matmul resulted in wrong answer.")
    end


    local fin_dcsf_val = fin_dcsf(M_dense, X, sol)
    if fin_dcsf_val != -1
        println("Density: ", inx / SIZE, "\tDCSF: ", fin_dcsf_val, "\tns")
        push!(pairs_fin_dcsf, (inx / SIZE, fin_dcsf_val))
    else
        println("Density: ", inx / SIZE, "\tDCSF: matmul resulted in wrong answer.")
    end


    local fin_coo_val = fin_coo(M_dense, X, sol)
    if fin_coo_val != -1
        println("Density: ", inx / SIZE, "\tCOO: ", fin_coo_val, "\tns")
        push!(pairs_fin_coo, (inx / SIZE, fin_coo_val))
    else
        println("Density: ", inx / SIZE, "\tCOO: matmul resulted in wrong answer.")
    end


    local fin_hash_val = fin_hash(M_dense, X, sol)
    if fin_hash_val != -1
        println("Density: ", inx / SIZE, "\tHash: ", fin_hash_val, "\tns")
        push!(pairs_fin_hash, (inx / SIZE, fin_hash_val))
    else
        println("Density: ", inx / SIZE, "\tHash: matmul resulted in wrong answer.")
    end


    local fin_bm_val = fin_bytemap(M_dense, X, sol)
    if fin_bm_val != -1
        println("Density: ", inx / SIZE, "\tBytemap: ", fin_bm_val, "\tns")
        push!(pairs_fin_bytemap, (inx / SIZE, fin_bm_val))
    else
        println("Density: ", inx / SIZE, "\tBytemap: matmul resulted in wrong answer.")
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
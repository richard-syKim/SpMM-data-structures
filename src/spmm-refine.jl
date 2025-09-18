# using Pkg
# Pkg.add("StatsBase")
# Pkg.add("BenchmarkTools")
# Pkg.add("JSON")
# Pkg.add("SuiteSparseGraphBLAS")
# Pkg.add("Finch")

using StatsBase
using BenchmarkTools
using JSON
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
end

function fin_DCSC(M, X, sol)
end

function fin_COO(M, X, sol)
end


inx = 0
inc = 1
cnt = 0
pairs_sa_csc = []
pairs_ssgblas = []
pairs_coo = []
while inx <= SIZE
    M_dense = create_sparse_mat(inx)
    X = randn(SIZE, SIZE)

    sol = M_dense * X

    local sa_csc_val = sa_csc(M_dense, X, sol)
    if sa_csc_val != -1
        println("Density: ", inx / SIZE, "\tCSC: ", sa_csc_val, "\tns")
        push!(pairs_csc, (inx / SIZE, sa_csc_val))
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



    if cnt >= 8
        global cnt = 0
        global inc *= 2
    end

    global inx += inc
    global cnt += 1
end

# open("results/results-naive.json", "w") do f
#     write(f, JSON.json(pairs_naive))
# end
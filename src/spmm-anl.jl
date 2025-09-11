using Pkg
Pkg.add("StatsBase")
Pkg.add("BenchmarkTools")
Pkg.add("JSON")
Pkg.add("SuiteSparseGraphBLAS")
Pkg.add("Finch")
Pkg.add("LinearAlgebra")
Pkg.add("CUDA")

using StatsBase
using BenchmarkTools
using JSON
using SparseArrays
using SuiteSparseGraphBLAS
using Finch
using CUDA
using CUDA.CUSPARSE

const SIZE = 4096

# s / SIZE is the density of the matrix
function sparseMatSetup(s, M = SIZE)
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

function customSparseMatSetup(s, M = SIZE)
    u = []

    # fill in s entries
    all_coords = collect(Iterators.product(1:M, 1:M))
    coords = sample(all_coords, s, replace=false)

    for coord in coords
        i, j = coord
        
        push!(u, (i, j, rand(Float64)))
    end

    return u
end

function customSparseMatMul(u, x, M = SIZE)
    y = zeros(M, M)

    for (i, j, v) in u
        for d in 1:M
            y[i, d] += v * x[j, d]
        end
    end

    return y
end


global i = 0
# global pairs_naive = []
# while i <= SIZE
#     m_naive = sparseMatSetup(i)
#     x = randn(SIZE, SIZE)

#     local bench_results = @benchmark $m_naive * $x
#     push!(pairs_naive, (i / SIZE, minimum(bench_results.times)))
#     println("Density: ", i / SIZE, "\tNaive: ", minimum(bench_results.times), "\tns")

#     global i += 16
# end

# open("results/results-naive.json", "w") do f
#     write(f, JSON.json(pairs_naive))
# end


# SparseArrays CSC format
# i = 0
# global pairs_csc = []
# while i <= SIZE
#     m_csc = sprand(SIZE, SIZE, i / SIZE)
#     x = randn(SIZE, SIZE)

#     local bench_results = @benchmark $m_csc * $x
#     push!(pairs_csc, (i / SIZE, minimum(bench_results.times)))
#     println("Density: ", i / SIZE, "\tCSC: ", minimum(bench_results.times), "\tns")

#     global i += 16
# end

# open("results/results-csc.json", "w") do f
#     write(f, JSON.json(pairs_csc))
# end

# i = 0
# global pairs_ssgblas = []
# while i <= SIZE
#     m_csc = sprand(SIZE, SIZE, i / SIZE)
#     x = randn(SIZE, SIZE)

#     m_SSGBLAS = GBMatrix(m_csc)
#     x_SSGBLAS = GBMatrix(x)

#     local bench_results = @benchmark $m_SSGBLAS * $x_SSGBLAS
#     push!(pairs_ssgblas, (i / SIZE, minimum(bench_results.times)))
#     println("Density: ", i / SIZE, "\tSSGBLAS: ", minimum(bench_results.times), "\tns")

#     global i += 16
# end

# open("results/results-ssgblas.json", "w") do f
#     write(f, JSON.json(pairs_ssgblas))
# end


# i = 0
# global pairs_custom = []
# while i <= SIZE
#     m_custom = customSparseMatSetup(i)
#     x = randn(SIZE, SIZE)

#     local bench_results = @benchmark customSparseMatMul($m_custom, $x)
#     push!(pairs_custom, (i / SIZE, minimum(bench_results.times)))
#     println("Density: ", i / SIZE, "\tcustom: ", minimum(bench_results.times), "\tns")

#     global i += 16
# end

# open("results/results-custom.json", "w") do f
#     write(f, JSON.json(pairs_custom))
# end


# Finch COO format
# i = 0
# global pairs_coo = []
# while i <= SIZE
#     m_coo = fsprand(Float64, (SIZE, SIZE), i / SIZE)
#     x = Dense(randn(SIZE, SIZE))

#     # TODO: may have to solve with custom multiply function?
#     local bench_results = @benchmark $m_coo * $x
#     push!(pairs_coo, (i / SIZE, minimum(bench_results.times)))
#     println("Density: ", i / SIZE, "\tcustom: ", minimum(bench_results.times), "\tns")

#     global i += 16
# end

# open("results/results-coo.json", "w") do f
#     write(f, JSON.json(pairs_coo))
# end


# # CUDA CSR format
# i = 0
# global pairs_cuda = []
# while i <= SIZE
#     m = sprand(SIZE, SIZE, i / SIZE)
#     x = randn(SIZE, SIZE)

#     m_cuda = CuSparseMatrixCSR(m)
#     x_cuda = CuArray(x)

#     local bench_results = @benchmark $m_cuda * $x_cuda
#     push!(pairs_cuda, (i / SIZE, minimum(bench_results.times)))
#     println("Density: ", i / SIZE, "\tcuda: ", minimum(bench_results.times), "\tns")

#     global i += 16
# end

# open("results/results-cuda.json", "w") do f
#     write(f, JSON.json(pairs_cuda))
# end


# CUDA CSR format
i = 0
global pairs_cuda = []
while i <= SIZE
    m = sprand(SIZE, SIZE, i / SIZE)
    x = randn(SIZE, SIZE)

    m_cuda = CuSparseMatrixCSR(m)
    x_cuda = CuArray(x)

    local bench_results = @benchmark $m_cuda * $x_cuda
    push!(pairs_cuda, (i / SIZE, minimum(bench_results.times)))
    println("Density: ", i / SIZE, "\tcuda: ", minimum(bench_results.times), "\tns")

    global i += 64
end

open("results/results-cuda.json", "w") do f
    write(f, JSON.json(pairs_cuda))
end
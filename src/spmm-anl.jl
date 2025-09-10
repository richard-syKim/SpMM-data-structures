using Pkg
Pkg.add("StatsBase")
Pkg.add("BenchmarkTools")
Pkg.add("JSON")

using StatsBase
using BenchmarkTools
using JSON
using SparseArrays

const SIZE = 4096


# s / SIZE is the density of the matrix
function sparseMatSetup(s, M = SIZE)
    # initialize u with random int
    u = zeros(M, M)

    # fill in M - s entries
    all_coords = collect(Iterators.product(1:M, 1:M))
    coords = sample(all_coords, s, replace=false)
    
    for coord in coords
        i, j = coord
        
        u[i, j] = rand(Float64)
    end

    return u
end


global i = 0
global pairs=  []
while i <= SIZE
    m_naive = sparseMatSetup(i)
    x = randn(SIZE, SIZE)
    local bench_results = @benchmark $m_naive * $x

    push!(pairs, (i / SIZE, minimum(bench_results.times)))
    println("Naive: \tDensity: ", i / SIZE, " Time: ", minimum(bench_results.times))


    m_csc = sprand(SIZE, SIZE, i / SIZE)
    bench_results = @benchmark $m_csc * $x
    println("CSC: \tDensity: ", i / SIZE, " Time: ", minimum(bench_results.times))

    global i += 16
end

open("spmm-anl_results-pls.json", "w") do f
    write(f, JSON.json(pairs))
end
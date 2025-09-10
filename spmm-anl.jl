using Pkg
Pkg.add("StatsBase")


Pkg.add("BenchmarkTools")

Pkg.add("JSON")

using StatsBase
using BenchmarkTools
using JSON
using SparseArrays


# function matMul(m, x)
#     return m*x
# end

# s / 4096 is the density of the matrix
function sparseMatSetup(s, M = 4096)
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

function vectSetup(M = 4096)
    x = zeros(M)
    for i in 1:M
        x[i] = rand(Float64)
    end

    return x
end


global i = 0
global pairs=  []
while i <= 4096
    m_naive = sparseMatSetup(4096 - i)
    x = sparseMatSetup(4096)
    local bench_results = @benchmark matMul($m, $x)

    push!(pairs, (i / 4096, minimum(bench_results.times)))
    println("Non-zeros: ", i / 4096, " Time: ", minimum(bench_results.times))


    m_csc = sprand(4096, 4096, i / 4096)

    global i += 16
end

open("spmm-anl_results-pls.json", "w") do f
    write(f, JSON.json(pairs))
end
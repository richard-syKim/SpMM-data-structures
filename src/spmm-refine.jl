using Pkg
Pkg.add("StatsBase")
Pkg.add("BenchmarkTools")
Pkg.add("SparseArrays")
Pkg.add("SuiteSparseGraphBLAS")
Pkg.add("Finch")
Pkg.add("JSON3")
Pkg.add("HDF5")

using StatsBase
using BenchmarkTools
using SparseArrays
using SuiteSparseGraphBLAS
using Finch
using JSON3
using HDF5


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


function main()
    N = 250
    eps = 1e-6

    indices = exp10.(range(log10(eps), log10(0.5), length=N))

    X = fread("ben/x.bsp.h5")

    # dense 
    pairs_dense = []
    println("For dense matrix multiplication:")
    for i in indices
        M_dense = fread(string("ben/m", i, ".bsp.h5"))
        X_dense = Array(X)
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

        println("\tDensity: ", i)
        M_dense = Array(M_dense)
        sol_calc = M_dense * X_dense

        bench_results = @benchmark $M_dense * $X_dense

        if isapprox(sol_calc, sol; rtol=1e-8, atol=1e-12)
            println("\t\tTime: \t", minimum(bench_results.times), "ns")
            push!(pairs_dense, (i, minimum(bench_results.times)))
        else
            println("\t\tWrong result.")
        end
    end

    open("res/0929/dense.json", "w") do f
        JSON3.write(f, pairs_dense)
    end


    # sparse arrays
    pairs_sa_csc = []
    println("For SparseArrays CSC multiplication:")
    for i in indices
        M_dense = fread(string("ben/m", i, ".bsp.h5"))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))
        X_dense = Array(X)

        println("\tDensity: ", i)
        M_dense = Array(M_dense)

        local sa_csc_val = sa_csc(M_dense, X_dense, sol)
        if sa_csc_val != -1
            println("\t\tTime: \t", sa_csc_val, "ns")
            push!(pairs_sa_csc, (i, sa_csc_val))
        else
            println("\t\tSparseArray CSC: matmul resulted in wrong answer.")
        end
    end

    open("res/0929/sparse-arrays-csc.json", "w") do f
        JSON3.write(f, pairs_sa_csc)
    end
end

main()
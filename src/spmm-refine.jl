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


const SIZE = 8192
const SKINNY = 100


function custom_coo_setup(M)
    # define the exact type of u to remove "Any"
    u = Vector{Tuple{Int, Int, Float64}}()

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
    # use @inbound to avoid bound check when accessing indices (y[i, d], X[j, d])
    @inbounds begin
        # @fastmath to allow floating point optimizations that are correct for real numbers
        @fastmath begin
            y = zeros(size(X, 1), size(X, 2))
            s = size(X, 2)
            for (i, j, v) in M
                for d in 1:s
                    y[i, d] += v * X[j, d]
                end
            end
        end
    end
    return y
end

function custom_coo_sep_mul(I, J, V, X)
    @inbounds begin
        @fastmath begin
            y = zeros(size(X, 1), size(X, 2))
            size_m = size(I, 1)
            s = size(X, 2)
            for p in 1:size_m
                i_p = I[p]
                j_p = J[p]
                v_p = V[p]

                # add potential SIMD
                @simd for d in 1:s
                    y[i_p, d] += v_p * X[j_p, d]
                end
            end
        end
    end
    return y
end


function custom_csc_mul(A::SparseMatrixCSC, X)
    @inbounds begin
        @fastmath begin
            # work with permuted matrix for row-major
            X_t = permutedims(X)
            ptr = A.colptr
            idx = A.rowval
            val = A.nzval

            m, n = size(A)
            n, d = size(X)

            Y_t = zeros(d, m)

            for j in 1:n
                st = ptr[j]
                ed = ptr[j+1] - 1
                for p in st:ed
                    i = idx[p]
                    v = val[p]
                    for k in 1:d
                        Y_t[k, i] += v * X_t[k, j]
                    end
                end
            end
        end
    end
    return permutedims(Y_t)
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


function main()
    N = 250
    eps = 1e-6

    indices = exp10.(range(log10(eps), log10(0.5), length=N))

    X = Array(fread("ben/x.bsp.h5"))

    # # dense 
    # pairs_dense = []
    # println("For dense matrix multiplication:")
    # for i in indices
    #     M_dense = fread(string("ben/m", i, ".bsp.h5"))
    #     sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

    #     println("\tDensity: ", i)
    #     M_dense = Array(M_dense)
    #     sol_calc = M_dense * X

    #     bench_results = @benchmark $M_dense * $X

    #     if isapprox(sol_calc, sol; rtol=1e-8, atol=1e-12)
    #         println("\t\tTime: \t", minimum(bench_results.times), "ns")
    #         push!(pairs_dense, (i, minimum(bench_results.times)))
    #     else
    #         println("\t\tWrong result.")
    #     end
    # end

    # open("res/0929/dense.json", "w") do f
    #     JSON3.write(f, pairs_dense)
    # end


    # # sparse arrays
    # pairs_sa_csc = []
    # println("For SparseArrays CSC multiplication:")
    # for i in indices
    #     M_dense = fread(string("ben/m", i, ".bsp.h5"))
    #     sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

    #     println("\tDensity: ", i)
    #     M_dense = Array(M_dense)

    #     local sa_csc_val = sa_csc(M_dense, X, sol)
    #     if sa_csc_val != -1
    #         println("\t\tTime: \t", sa_csc_val, "ns")
    #         push!(pairs_sa_csc, (i, sa_csc_val))
    #     else
    #         println("\t\tSparseArray CSC: matmul resulted in wrong answer.")
    #     end
    # end

    # open("res/0929/sparse-arrays-csc.json", "w") do f
    #     JSON3.write(f, pairs_sa_csc)
    # end


    # custom coo
    pairs_coo = []
    println("For custom coo multiplication:")
    k = -1
    for i in indices
        # reduce number of runs
        k += 1
        if k % 10 != 0
            continue
        end

        M_dense = fread(string("ben/m", i, ".bsp.h5"))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

        println("\tDensity: ", i)
        M_dense = Array(M_dense)

        local custom_coo_val = custom_coo(M_dense, X, sol)
        if custom_coo_val != -1
            println("\t\tTime: \t", custom_coo_val, "ns")
            push!(pairs_coo, (i, custom_coo_val))
        else
            println("\t\tcustom COO: matmul resulted in wrong answer.")
        end
    end

    open("res/1001/custom-coo-opt.json", "w") do f
        JSON3.write(f, pairs_coo)
    end


end

main()
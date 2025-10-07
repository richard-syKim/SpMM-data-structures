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


function custom_coo_setup(M)
    # define the exact type of u to remove "Any"
    u = Vector{Tuple{Int, Int, Float64}}()

    for i in 1:size(M, 1)
        for j in 1:size(M, 2)
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


function unzip(A)
    return map(x->getfield.(A, x), fieldnames(eltype(A)))
end


function custom_coo_sep(M, X, sol)
    m_coo = custom_coo_setup(M)
    I, J, V = unzip(m_coo)

    bench_results = @benchmark custom_coo_sep_mul($I, $J, $V, $X)
    temp = custom_coo_sep_mul(I, J, V, X)
    if isapprox(temp, sol; rtol=1e-8, atol=1e-12)
        return (minimum(bench_results.times))
    else
        return -1
    end
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


function custom_csc(M, X, sol)
    m_csc = sparse(M)

    bench_results = @benchmark custom_csc_mul($m_csc, $X)
    temp = custom_csc_mul(m_csc, X)
    if isapprox(temp, sol; rtol=1e-8, atol=1e-12)
        return (minimum(bench_results.times))
    else
        return -1
    end
end


function bytemap(M)
    B = zeros(Int8, size(M, 1), size(M, 2))
    val = Vector{Float64}()
    for i in size(M, 1)
        for j in size(M, 2)
            if M[i, j] != 0.0
                B[i, j] = 1
                push!(val, M[i, j])
            else
                B[i, j] = 0
            end
        end
    end
    return B, val 
end

function custom_bytemap(M, X, sol)
    B, val = bytemap(M)
    return 
end


function main()
    N = 250
    eps = 1e-6

    indices = exp10.(range(log10(eps), log10(0.5), length=N))

    X = fread("ben/x.bsp.h5")


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
        X_dense = Array(X)

        println("\tDensity: ", i)
        M_dense = Array(M_dense)

        local custom_coo_val = custom_coo(M_dense, X_dense, sol)
        if custom_coo_val != -1
            println("\t\tTime: \t", custom_coo_val, "ns")
            push!(pairs_coo, (i, custom_coo_val))
        else
            println("\t\tcustom COO: matmul resulted in wrong answer.")
        end
    end

    open("res/1005/custom-coo-opt.json", "w") do f
        JSON3.write(f, pairs_coo)
    end


    # custom coo with decoupled tuples
    pairs_coo_sep = []
    println("For custom coo with decoupled tuples multiplication:")
    k = -1
    for i in indices
        # reduce number of runs
        k += 1
        if k % 10 != 0
            continue
        end

        M_dense = fread(string("ben/m", i, ".bsp.h5"))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))
        X_dense = Array(X)

        println("\tDensity: ", i)
        M_dense = Array(M_dense)

        local custom_coo_sep_val = custom_coo_sep(M_dense, X_dense, sol)
        if custom_coo_sep_val != -1
            println("\t\tTime: \t", custom_coo_sep_val, "ns")
            push!(pairs_coo_sep, (i, custom_coo_sep_val))
        else
            println("\t\tcustom COO sep: matmul resulted in wrong answer.")
        end
    end

    open("res/1005/custom-coo-sep.json", "w") do f
        JSON3.write(f, pairs_coo_sep)
    end


    # custom csc
    pairs_csc = []
    println("For custom csc multiplication:")
    # k = -1
    for i in indices
        # # reduce number of runs
        # k += 1
        # if k % 10 != 0
        #     continue
        # end

        M_dense = fread(string("ben/m", i, ".bsp.h5"))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))
        X_dense = Array(X)

        println("\tDensity: ", i)
        M_dense = Array(M_dense)

        local custom_csc_val = custom_csc(M_dense, X_dense, sol)
        if custom_csc_val != -1
            println("\t\tTime: \t", custom_csc_val, "ns")
            push!(pairs_csc, (i, custom_csc_val))
        else
            println("\t\tcustom CSC: matmul resulted in wrong answer.")
        end
    end

    open("res/1005/custom-csc.json", "w") do f
        JSON3.write(f, pairs_csc)
    end

end

main()
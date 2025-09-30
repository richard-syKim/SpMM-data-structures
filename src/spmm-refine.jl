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


const SIZE = 4096


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

function finch_isapprox(a, b; rtol=1e-8, atol=1e-12)
    @finch begin
        for i in _
            for j in _
                if !(abs(a[i, j] - b[i, j]) <= atol + rtol * max(abs(a[i, j]), abs(b[i, j])))
                    return false
                end
            end
        end
    end
    return true
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


function main()
    N = 250
    eps = 1e-6

    indices = exp10.(range(log10(eps), log10(0.5), length=N))

    X = Array(fread("ben/x.bsp.h5"))

    # dense 
    pairs_dense = []
    println("For dense matrix multiplication:")
    for i in indices
        M_dense = Array(fread(string("ben/m", i, ".bsp.h5")))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

        println("\tDensity: ", i)
        sol_calc = M_dense * X

        bench_results = @benchmark $M_dense * $X

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
        M_dense = Array(fread(string("ben/m", i, ".bsp.h5")))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

        println("\tDensity: ", i)

        local sa_csc_val = sa_csc(M_dense, X, sol)
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


    # custom coo
    pairs_coo = []
    for i in indices
        M_dense = Array(fread(string("ben/m", i, ".bsp.h5")))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

        println("Density: ", i)

        local custom_coo_val = custom_coo(M_dense, X, sol)
        if custom_coo_val != -1
            println("\tcustom COO: \t\t", custom_coo_val, "ns")
            push!(pairs_coo, (i, custom_coo_val))
        else
            println("\tcustom COO: matmul resulted in wrong answer.")
        end
    end

    open("res/0929/custom-coo.json", "w") do f
        JSON3.write(f, pairs_coo)
    end


    # finch csc 
    pairs_fin_csc = []
    for i in indices
        M_dense = Array(fread(string("ben/m", i, ".bsp.h5")))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

        println("Density: ", i)

        local fin_csc_val = fin_csc(M_dense, X, sol)
        if fin_csc_val != -1
            println("\tfinch CSC: \t\t", fin_csc_val, "ns")
            push!(pairs_fin_csc, (i, fin_csc_val))
        else
            println("\tfinch CSC: matmul resulted in wrong answer.")
        end
    end

    open("res/0929/finch-csc.json", "w") do f
        JSON3.write(f, pairs_fin_csc)
    end


    # finch csf
    pairs_fin_csf = []
    for i in indices
        M_dense = Array(fread(string("ben/m", i, ".bsp.h5")))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

        println("Density: ", i)

        local fin_csf_val = fin_csf(M_dense, X, sol)
        if fin_csf_val != -1
            println("\tfinch CSF: \t\t", fin_csf_val, "ns")
            push!(pairs_fin_csf, (i, fin_csf_val))
        else
            println("\tfinch CSF: matmul resulted in wrong answer.")
        end
    end

    open("res/0929/finch-csf.json", "w") do f
        JSON3.write(f, pairs_fin_csf)
    end


    # finch dcsc
    pairs_fin_dcsc = []
    for i in indices
        M_dense = Array(fread(string("ben/m", i, ".bsp.h5")))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

        println("Density: ", i)

        local fin_dcsc_val = fin_dcsc(M_dense, X, sol)
        if fin_dcsc_val != -1
            println("\tfinch DCSC: \t\t", fin_dcsc_val, "ns")
            push!(pairs_fin_dcsc, (i, fin_dcsc_val))
        else
            println("\tfinch DCSC: matmul resulted in wrong answer.")
        end
    end

    open("res/0929/finch-dcsc.json", "w") do f
        JSON3.write(f, pairs_fin_dcsc)
    end


    # finch dcsf
    pairs_fin_dcsf = []
    for i in indices
        M_dense = Array(fread(string("ben/m", i, ".bsp.h5")))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

        println("Density: ", i)

        local fin_dcsf_val = fin_dcsf(M_dense, X, sol)
        if fin_dcsf_val != -1
            println("\tfinch DCSF: \t\t", fin_dcsf_val, "ns")
            push!(pairs_fin_dcsf, (i, fin_dcsf_val))
        else
            println("\tfinch DCSF: matmul resulted in wrong answer.")
        end
    end

    open("res/0929/finch-dcsf.json", "w") do f
        JSON3.write(f, pairs_fin_dcsf)
    end


    # finch coo
    pairs_fin_coo = []
    for i in indices
        M_dense = Array(fread(string("ben/m", i, ".bsp.h5")))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

        println("Density: ", i)

        local fin_coo_val = fin_coo(M_dense, X, sol)
        if fin_coo_val != -1
            println("\tfinch COO: \t\t", fin_coo_val, "ns")
            push!(pairs_fin_coo, (i, fin_coo_val))
        else
            println("\tfinch COO: matmul resulted in wrong answer.")
        end
    end

    open("res/0929/finch-coo.json", "w") do f
        JSON3.write(f, pairs_fin_coo)
    end


    # finch hash
    pairs_fin_hash = []
    for i in indices
        M_dense = Array(fread(string("ben/m", i, ".bsp.h5")))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

        println("Density: ", i)

        local fin_hash_val = fin_hash(M_dense, X, sol)
        if fin_hash_val != -1
            println("\tfinch Hash: \t\t", fin_hash_val, "ns")
            push!(pairs_fin_hash, (i, fin_hash_val))
        else
            println("\tfinch Hash: matmul resulted in wrong answer.")
        end
    end

    open("res/0929/finch-hash.json", "w") do f
        JSON3.write(f, pairs_fin_hash)
    end


    # finch bytemap
    pairs_fin_bytemap = []
    for i in indices
        M_dense = Array(fread(string("ben/m", i, ".bsp.h5")))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

        println("Density: ", i)

        local fin_bm_val = fin_bytemap(M_dense, X, sol)
        if fin_bm_val != -1
            println("\tfinch Bytemap: \t\t", fin_bm_val, "ns")
            push!(pairs_fin_bytemap, (i, fin_bm_val))
        else
            println("\tfinch Bytemap: matmul resulted in wrong answer.")
        end
    end

    open("res/0929/finch-bm.json", "w") do f
        JSON3.write(f, pairs_fin_bytemap)
    end
end

main()
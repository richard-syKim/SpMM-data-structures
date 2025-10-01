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





function fin_csc(M, X, sol)
    m_ten = Finch.Tensor(CSCFormat(), M)
    x_ten = Finch.Tensor(CSCFormat(), X)
    temp = zeros(SIZE, SKINNY)

    t = @benchmark begin 
        $temp .= 0.0
        @finch begin
            for k = _
                for j = _
                    for i = _
                        $temp[i, k] += $m_ten[i, j] * $x_ten[j, k]
                    end
                end
            end
        end
    end

    @finch begin
        temp .= 0.0
        for k = _
            for j = _
                for i = _
                    temp[i, k] += m_ten[i, j] * x_ten[j, k]
                end
            end
        end
    end

    if isapprox(temp, sol; rtol=1e-8, atol=1e-12)
        return minimum(t.times)
    else
        return -1
    end
end

function fin_csf(M, X, sol)
    m_ten = Finch.Tensor(CSFFormat(2), M)
    x_ten = Finch.Tensor(CSFFormat(2), X)
    temp = zeros(SIZE, SKINNY)

    t = @benchmark begin 
        $temp .= 0.0
        @finch begin
            for k = _
                for j = _
                    for i = _
                        $temp[i, k] += $m_ten[i, j] * $x_ten[j, k]
                    end
                end
            end
        end
    end

    @finch begin
        temp .= 0.0
        for k = _
            for j = _
                for i = _
                    temp[i, k] += m_ten[i, j] * x_ten[j, k]
                end
            end
        end
    end

    if isapprox(temp, sol; rtol=1e-8, atol=1e-12)
        return minimum(t.times)
    else
        return -1
    end
end

function fin_dcsc(M, X, sol)
    m_ten = Finch.Tensor(DCSCFormat(), M)
    x_ten = Finch.Tensor(DCSCFormat(), X)
    temp = zeros(SIZE, SKINNY)

    t = @benchmark begin 
        $temp .= 0.0
        @finch begin
            for k = _
                for j = _
                    for i = _
                        $temp[i, k] += $m_ten[i, j] * $x_ten[j, k]
                    end
                end
            end
        end
    end

    @finch begin
        temp .= 0.0
        for k = _
            for j = _
                for i = _
                    temp[i, k] += m_ten[i, j] * x_ten[j, k]
                end
            end
        end
    end

    if isapprox(temp, sol; rtol=1e-8, atol=1e-12)
        return minimum(t.times)
    else
        return -1
    end
end

function fin_dcsf(M, X, sol)
    m_ten = Finch.Tensor(DCSFFormat(2), M)
    x_ten = Finch.Tensor(DCSFFormat(2), X)
    temp = zeros(SIZE, SKINNY)

    t = @benchmark begin 
        $temp .= 0.0
        @finch begin
            for k = _
                for j = _
                    for i = _
                        $temp[i, k] += $m_ten[i, j] * $x_ten[j, k]
                    end
                end
            end
        end
    end

    @finch begin
        temp .= 0.0
        for k = _
            for j = _
                for i = _
                    temp[i, k] += m_ten[i, j] * x_ten[j, k]
                end
            end
        end
    end

    if isapprox(temp, sol; rtol=1e-8, atol=1e-12)
        return minimum(t.times)
    else
        return -1
    end
end

function fin_coo(M, X, sol)
    m_ten = Finch.Tensor(COOFormat(2), M)
    x_ten = Finch.Tensor(COOFormat(2), X)
    temp = zeros(SIZE, SKINNY)

    t = @benchmark begin 
        $temp .= 0.0
        @finch begin
            for k = _
                for j = _
                    for i = _
                        $temp[i, k] += $m_ten[i, j] * $x_ten[j, k]
                    end
                end
            end
        end
    end

    @finch begin
        temp .= 0.0
        for k = _
            for j = _
                for i = _
                    temp[i, k] += m_ten[i, j] * x_ten[j, k]
                end
            end
        end
    end

    if isapprox(temp, sol; rtol=1e-8, atol=1e-12)
        return minimum(t.times)
    else
        return -1
    end
end

function fin_hash(M, X, sol)
    m_ten = Finch.Tensor(HashFormat(2), M)
    x_ten = Finch.Tensor(HashFormat(2), X)
    temp = zeros(SIZE, SKINNY)

    t = @benchmark begin 
        $temp .= 0.0
        @finch begin
            for k = _
                for j = _
                    for i = _
                        $temp[i, k] += $m_ten[i, j] * $x_ten[j, k]
                    end
                end
            end
        end
    end

    @finch begin
        temp .= 0.0
        for k = _
            for j = _
                for i = _
                    temp[i, k] += m_ten[i, j] * x_ten[j, k]
                end
            end
        end
    end

    if isapprox(temp, sol; rtol=1e-8, atol=1e-12)
        return minimum(t.times)
    else
        return -1
    end
end

function fin_bytemap(M, X, sol)
    m_ten = Finch.Tensor(ByteMapFormat(2), M)
    x_ten = Finch.Tensor(ByteMapFormat(2), X)
    temp = zeros(SIZE, SKINNY)

    t = @benchmark begin 
        $temp .= 0.0
        @finch begin
            for k = _
                for j = _
                    for i = _
                        $temp[i, k] += $m_ten[i, j] * $x_ten[j, k]
                    end
                end
            end
        end
    end

    @finch begin
        temp .= 0.0
        for k = _
            for j = _
                for i = _
                    temp[i, k] += m_ten[i, j] * x_ten[j, k]
                end
            end
        end
    end

    if isapprox(temp, sol; rtol=1e-8, atol=1e-12)
        return minimum(t.times)
    else
        return -1
    end
end


function main()
    N = 250
    eps = 1e-6

    indices = exp10.(range(log10(eps), log10(0.5), length=N))

    X = Array(fread("ben/x.bsp.h5"))


    # finch csc 
    pairs_fin_csc = []
    println("For finch csc multiplication:")
    for i in indices
        M_dense = fread(string("ben/m", i, ".bsp.h5"))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

        println("\tDensity: ", i)
        M_dense = Array(M_dense)

        local fin_csc_val = fin_csc(M_dense, X, sol)
        if fin_csc_val != -1
            println("\t\tTime: \t", fin_csc_val, "ns")
            push!(pairs_fin_csc, (i, fin_csc_val))
        else
            println("\t\tfinch CSC: matmul resulted in wrong answer.")
        end
    end

    open("res/1001/finch-csc.json", "w") do f
        JSON3.write(f, pairs_fin_csc)
    end


    # finch csf
    pairs_fin_csf = []
    println("For finch csf multiplication:")
    for i in indices
                M_dense = fread(string("ben/m", i, ".bsp.h5"))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

        println("\tDensity: ", i)
        M_dense = Array(M_dense)

        local fin_csf_val = fin_csf(M_dense, X, sol)
        if fin_csf_val != -1
            println("\t\tTime: \t", fin_csf_val, "ns")
            push!(pairs_fin_csf, (i, fin_csf_val))
        else
            println("\t\tfinch CSF: matmul resulted in wrong answer.")
        end
    end

    open("res/1001/finch-csf.json", "w") do f
        JSON3.write(f, pairs_fin_csf)
    end


    # finch dcsc
    pairs_fin_dcsc = []
    println("For finch dcsc multiplication:")
    for i in indices
                M_dense = fread(string("ben/m", i, ".bsp.h5"))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

        println("\tDensity: ", i)
        M_dense = Array(M_dense)

        local fin_dcsc_val = fin_dcsc(M_dense, X, sol)
        if fin_dcsc_val != -1
            println("\t\tTime: \t", fin_dcsc_val, "ns")
            push!(pairs_fin_dcsc, (i, fin_dcsc_val))
        else
            println("\t\tfinch DCSC: matmul resulted in wrong answer.")
        end
    end

    open("res/1001/finch-dcsc.json", "w") do f
        JSON3.write(f, pairs_fin_dcsc)
    end


    # finch dcsf
    pairs_fin_dcsf = []
    println("For finch dcsf multiplication:")
    for i in indices
                M_dense = fread(string("ben/m", i, ".bsp.h5"))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

        println("\tDensity: ", i)
        M_dense = Array(M_dense)

        local fin_dcsf_val = fin_dcsf(M_dense, X, sol)
        if fin_dcsf_val != -1
            println("\tfinch DCSF: \t\t", fin_dcsf_val, "ns")
            push!(pairs_fin_dcsf, (i, fin_dcsf_val))
        else
            println("\tfinch DCSF: matmul resulted in wrong answer.")
        end
    end

    open("res/1001/finch-dcsf.json", "w") do f
        JSON3.write(f, pairs_fin_dcsf)
    end


    # finch coo
    pairs_fin_coo = []
    println("For finch coo multiplication:")
    for i in indices
                M_dense = fread(string("ben/m", i, ".bsp.h5"))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

        println("\tDensity: ", i)
        M_dense = Array(M_dense)

        local fin_coo_val = fin_coo(M_dense, X, sol)
        if fin_coo_val != -1
            println("\tfinch COO: \t\t", fin_coo_val, "ns")
            push!(pairs_fin_coo, (i, fin_coo_val))
        else
            println("\tfinch COO: matmul resulted in wrong answer.")
        end
    end

    open("res/1001/finch-coo.json", "w") do f
        JSON3.write(f, pairs_fin_coo)
    end


    # finch hash
    pairs_fin_hash = []
    println("For finch hash multiplication:")
    for i in indices
                M_dense = fread(string("ben/m", i, ".bsp.h5"))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

        println("\tDensity: ", i)
        M_dense = Array(M_dense)

        local fin_hash_val = fin_hash(M_dense, X, sol)
        if fin_hash_val != -1
            println("\tfinch Hash: \t\t", fin_hash_val, "ns")
            push!(pairs_fin_hash, (i, fin_hash_val))
        else
            println("\tfinch Hash: matmul resulted in wrong answer.")
        end
    end

    open("res/1001/finch-hash.json", "w") do f
        JSON3.write(f, pairs_fin_hash)
    end


    # finch bytemap
    pairs_fin_bytemap = []
    println("For finch bytemap multiplication:")
    for i in indices
                M_dense = fread(string("ben/m", i, ".bsp.h5"))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))

        println("\tDensity: ", i)
        M_dense = Array(M_dense)

        local fin_bm_val = fin_bytemap(M_dense, X, sol)
        if fin_bm_val != -1
            println("\tfinch Bytemap: \t\t", fin_bm_val, "ns")
            push!(pairs_fin_bytemap, (i, fin_bm_val))
        else
            println("\tfinch Bytemap: matmul resulted in wrong answer.")
        end
    end

    open("res/1001/finch-bm.json", "w") do f
        JSON3.write(f, pairs_fin_bytemap)
    end
end

main()
using Pkg
Pkg.add("StatsBase")
Pkg.add("Finch")
Pkg.add("HDF5")
Pkg.add("BenchmarkTools")
Pkg.add("JSON3")

using StatsBase
using Finch
using HDF5
using BenchmarkTools
using JSON3

const SIZE = 8192
const SKINNY = 100


function isstruct(x)
    return x isa Any && !(x isa Number) && !(x isa AbstractArray)
end

function finch_isapprox(a, b; rtol=1e-8, atol=1e-12)
    if typeof(a) != typeof(b)
        # println("type diff: ", typeof(a), " vs ", typeof(b))
        return false
    end

    if a isa Float64 && b isa Float64
        return abs(a - b) <= atol + rtol * max(abs(a), abs(b))
    elseif a isa Integer && b isa Integer
        return a == b
    elseif a isa AbstractArray && b isa AbstractArray
        if size(a) != size(b)
            # println("size diff: ", size(a), " vs ", size(b))
            return false
        end

        for (x, y) in zip(a, b)
            if !finch_isapprox(x, y; rtol=rtol, atol=atol)
                # println("element diff: ", x, " vs ", y)
                return false
            end
        end
        return true

    elseif a isa Finch.Tensor && b isa Finch.Tensor
        if size(a) != size(b)
            # println("tensor size diff: ", size(a), " vs ", size(b))
            return false
        end
        for I in CartesianIndices(size(a))
            av = a[I]
            bv = b[I]
            if !(abs(av - bv) <= atol + rtol * max(abs(av), abs(bv)))
                # println("tensor value diff at $I: $av vs $bv")
                return false
            end
        end
        return true

    elseif isstruct(a) && isstruct(b)
        for name in fieldnames(typeof(a))
            ax = getfield(a, name)
            bx = getfield(b, name)
            if !finch_isapprox(ax, bx; rtol=rtol, atol=atol)
                # println("field ", name, " diff: ", ax, " vs ", bx)
                return false
            end
        end
        return true
    else
        return isequal(a, b)
    end
end

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


function main()
    N = 250
    eps = 1e-6

    indices = exp10.(range(log10(eps), log10(0.5), length=N))

    X = Array(fread("ben/x.bsp.h5"))

 
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

    open("res/0930/finch-csc.json", "w") do f
        JSON3.write(f, pairs_fin_csc)
    end
end

main()
using Pkg
Pkg.add("StatsBase")
Pkg.add("Finch")
Pkg.add("JSON3")
Pkg.add("BenchmarkTools")

using StatsBase
using Finch
using JSON3
using BenchmarkTools


const SIZE = 4096


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
    sol_ten = Finch.Tensor(CSCFormat(), sol)
    temp = Finch.Tensor(CSCFormat(), zeros(Float64, SIZE, SIZE))
    # temp = zeros(Float64, SIZE, SIZE)

    # @finch begin
    #     for i in _
    #         let col_sum = 0.0
    #             for j in m_ten.data.data.ptrs[i]:(m_ten.data.data.ptrs[i+1]-1)
    #                 col_sum = col_sum + m_ten.data.data.values.values[j]
    #             end

    #             for j in _
    #                 temp[i, j] = col_sum * X[i, j]
    #             end
    #         end
            
    #     end
    # end


    bench_results = @benchmark @finch begin
            for i in _
                for j in _
                    for l in _
                        temp[i, l] += m_ten[i, j] * x_ten[j, l]
                    end
                end
            end
            # return temp
        end

    temp_ten = Finch.Tensor(CSCFormat(), temp)
    if finch_isapprox(temp_ten, sol_ten; rtol=1e-8, atol=1e-12)
        return (minimum(bench_results.times))
    else
        return -1
    end
end


inx = 0
inc = 1
cnt = 0
pairs_fin_csc = []

# X = JSON3.read("ben/x.json", Vector{Vector{Float64}})
# X = reduce(hcat, X) |> permutedims

X = rand(Float64, SIZE, SIZE)


while inx <= (SIZE / 2)
    # M_dense = JSON3.read(string("ben/m", inx, ".json"), Vector{Vector{Float64}})
    # M_dense = reduce(hcat, M_dense) |> permutedims

    M_dense = randn(Float64, SIZE, SIZE)

    # sol = JSON3.read(string("ben/sol", inx,".json"), Vector{Vector{Float64}})
    # sol = reduce(hcat, sol) |> permutedims

    sol = M_dense * X

    println("Density: ", inx / SIZE)

    local fin_csc_val = fin_csc(M_dense, X, sol)
    if fin_csc_val != -1
        println("\tfinch CSC: \t\t", fin_csc_val, "ns")
        push!(pairs_fin_csc, (inx / SIZE, fin_csc_val))
    else
        println("\tfinch CSC: matmul resulted in wrong answer.")
    end

    if cnt >= 16
        global cnt = 0
        global inc *= 2
    end

    global inx += inc
    global cnt += 1
end

open("res/finch-csc.json", "w") do f
    write(f, JSON.json(pairs_fin_csc))
end
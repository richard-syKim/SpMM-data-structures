using Pkg
Pkg.add("StatsBase")
Pkg.add("JSON3")
Pkg.add("Finch")
Pkg.add("HDF5")

using StatsBase
using JSON3
using Finch
using HDF5

const SIZE = 8192

function main()
    X = Tensor(Dense(SparseList(Element(0.0))), randn(SIZE, 100))

    fwrite("ben/x.bsp.h5", X)

    N = 250
    eps = 1e-6

    indices = exp10.(range(log10(eps), log10(0.5), length=N))

    for i in indices
        println("Wrote ben/m",  i, ".bsp.h5 and ben/sol", i, ".bsp.h5")
        
        M_dense = fsprand(Float64, SIZE, SIZE, i)

        sol = Tensor(Dense(SparseList(Element(0.0))), M_dense * X)

        fwrite(string("ben/m",  i, ".bsp.h5"), M_dense)
        fwrite(string("ben/sol", i, ".bsp.h5"), sol)
    end
end

main()
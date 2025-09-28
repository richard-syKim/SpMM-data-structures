using Pkg
Pkg.add("StatsBase")
Pkg.add("JSON3")

using StatsBase
using JSON3

const SIZE = 4096 * 2

# s is the density of the matrix
function create_sparse_mat(s, M = SIZE)

    return sprand(Float64, M, M, s)
end

inx = 0
inc = 1
cnt = 0
X = randn(SIZE, SIZE)

while inx <= (SIZE / 2)
    M_dense = create_sparse_mat(inx)

    sol = M_dense * X

    JSON3.write(string("ben/m",  inx, ".json"), collect(eachrow(M_dense)))

    JSON3.write(string("ben/sol", inx, ".json"), collect(eachrow(sol)))

    if cnt >= 16
        global cnt = 0
        global inc *= 2
    end

    global inx += inc
    global cnt += 1
end

JSON3.write("ben/x.json", collect(eachrow(X)))
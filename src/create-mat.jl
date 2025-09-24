using Pkg
Pkg.add("StatsBase")
Pkg.add("JSON")

using StatsBase
using JSON

const SIZE = 4096

# s / SIZE is the density of the matrix
function create_sparse_mat(s, M = SIZE)
    # initialize u with random int
    u = zeros(M, M)

    # fill in s entries
    all_coords = collect(Iterators.product(1:M, 1:M))
    coords = sample(all_coords, s, replace=false)
    
    for coord in coords
        i, j = coord
        
        u[i, j] = rand(Float64)
    end

    return u
end

inx = 0
inc = 1
cnt = 0
X = randn(SIZE, SIZE)

while inx <= (SIZE / 2)
    M_dense = create_sparse_mat(inx)

    sol = M_dense * X


# sparse matrix -> store triplets (i, j, value)
    open(string("ben/m",inx, ".json"), "w") do f
        write(f, JSON.json(M_dense))
    end

    open(string("ben/sol", inx,".json"), "w") do f
        write(f, JSON.json(sol))
    end


    if cnt >= 16
        global cnt = 0
        global inc *= 2
    end

    global inx += inc
    global cnt += 1
end

open("ben/x.json", "w") do f
        write(f, JSON.json(X))
    end
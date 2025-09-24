# import Pkg; Pkg.add("Plots")

# Pkg.activate()
# Pkg.add("BenchmarkTools")
# using BenchmarkTools

# Pkg.add("Profile")
# using Profile

# @benchmark function lap2d!(u, unew)
#     M, N = size(u)
#     for j in 2:N-1
#         for i in 2:M-1
#             unew[i, j] = 0.25 * (u[i + 1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])
#         end
#     end
# end

# function setup(N=4096, M=4096)
#     u = zeros(M, N)
#     # set boundary conditions
#     u[1,:] = u[end,:] = u[:,1] = u[:,end] .= 10.0
#     unew = copy(u);
#     return u, unew
# end

# u, unew = setup()

# for i in 1:1000
#     lap2d!(u, unew)

#     global u = copy(unew)

# end

# using Plots
# heatmap(u)

# bench_results = @benchmark lap2d!(u, unew)

# typeof(bench_results)

# println(minimum(bench_results.times))


# Profile.clear()
# @profile lap2d!(u, unew)
# Profile.print()


# using Pkg
# Pkg.add("Finch")

# using Finch


# function isstruct(x)
#     return x isa Any && !(x isa Number) && !(x isa AbstractArray)
# end


# function finch_isapprox(a, b; rtol=1e-8, atol=1e-12)
#     if typeof(a) != typeof(b)
#         println("type diff: ", typeof(a), " vs ", typeof(b))
#         return false
#     end

#     if a isa Float64 && b isa Float64
#         return abs(a - b) ≤ atol + rtol * max(abs(a), abs(b))
#     elseif a isa Integer && b isa Integer
#         return a == b
#     elseif a isa AbstractArray && b isa AbstractArray
#         if size(a) != size(b)
#             println("size diff: ", size(a), " vs ", size(b))
#             return false
#         end

#         for (x, y) in zip(a, b)
#             if !finch_isapprox(x, y; rtol=rtol, atol=atol)
#                 println("element diff: ", x, " vs ", y)
#                 return false
#             end
#         end
#         return true

#     elseif a isa Finch.Tensor && b isa Finch.Tensor
#         if size(a) != size(b)
#             println("tensor size diff: ", size(a), " vs ", size(b))
#             return false
#         end
#         for I in CartesianIndices(size(a))
#             av = a[I]
#             bv = b[I]
#             if !(abs(av - bv) ≤ atol + rtol * max(abs(av), abs(bv)))
#                 println("tensor value diff at $I: $av vs $bv")
#                 return false
#             end
#         end
#         return true

#     elseif isstruct(a) && isstruct(b)
#         for name in fieldnames(typeof(a))
#             ax = getfield(a, name)
#             bx = getfield(b, name)
#             if !finch_isapprox(ax, bx; rtol=rtol, atol=atol)
#                 println("field ", name, " diff: ", ax, " vs ", bx)
#                 return false
#             end
#         end
#         return true
#     else
#         return isequal(a, b)
#     end
# end



# A = Finch.Tensor(HashFormat(2), [1.0 0; 0 -1.0])

# B = Finch.Tensor(HashFormat(2), [1.0 0; 0 -1.0])

# C = Finch.Tensor(HashFormat(2), [1.0 0; 0 1.0])

# D = A * C

# E = Finch.Tensor(HashFormat(2), D)

# # F = Sparse(E)

# println(A)
# println(E)
# println(typeof(A))
# println(typeof(E))
# println(finch_isapprox(A, B; rtol=1e-8, atol=1e-12))
# println(finch_isapprox(A, D; rtol=1e-8, atol=1e-12))
# println(finch_isapprox(A, E; rtol=1e-8, atol=1e-12))


using Pkg
Pkg.add("StatsBase")
Pkg.add("JSON3")

using StatsBase
using JSON3

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
X = [1 0; 1 0]

X_nested = collect(eachrow(X))

JSON3.write("x.json", X_nested)
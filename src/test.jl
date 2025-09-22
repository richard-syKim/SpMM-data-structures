import Pkg; Pkg.add("Plots")

Pkg.activate()
Pkg.add("BenchmarkTools")
using BenchmarkTools

Pkg.add("Profile")
using Profile

@benchmark function lap2d!(u, unew)
    M, N = size(u)
    for j in 2:N-1
        for i in 2:M-1
            unew[i, j] = 0.25 * (u[i + 1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])
        end
    end
end

function setup(N=4096, M=4096)
    u = zeros(M, N)
    # set boundary conditions
    u[1,:] = u[end,:] = u[:,1] = u[:,end] .= 10.0
    unew = copy(u);
    return u, unew
end

u, unew = setup()

# for i in 1:1000
#     lap2d!(u, unew)

#     global u = copy(unew)

# end

# using Plots
# heatmap(u)

# bench_results = @benchmark lap2d!(u, unew)

# typeof(bench_results)

# println(minimum(bench_results.times))


Profile.clear()
@profile lap2d!(u, unew)
Profile.print()


# using Pkg
# Pkg.add("Finch")

# using Finch


# function isstruct(x)
#     return x isa Any && !(x isa Number) && !(x isa AbstractArray)
# end


# function finch_isapprox(a, b; rtol=1e-8, atol=1e-12)
#     if typeof(a) != typeof(b)
#         return false
#     end

#     if a isa Float64 && b isa Float64
#         return abs(a - b) ≤ atol + rtol * max(abs(a), abs(b))
#     elseif a isa Integer && b isa Integer
#         return a == b
#     elseif a isa AbstractArray && b isa AbstractArray
#         if size(a) != size(b)
#             return false
#         end

#         for (x, y) in zip(a, b)
#             if !finch_isapprox(x, y; rtol=rtol, atol=atol)
#                 return false
#             end
#         end
#         return true

#     elseif isstruct(a) && isstruct(b)
#         for name in fieldnames(typeof(a))
#             ax = getfield(a, name)
#             bx = getfield(b, name)
#             if !finch_isapprox(ax, bx; rtol=rtol, atol=atol)
#                 return false
#             end
#         end
#         return true

#     else
#         return isequal(a, b)
#     end
# end



# A = Finch.Tensor(CSCFormat(), [1.0 0; 0 -1.0])

# B = Finch.Tensor(CSCFormat(), [1.0 0; 0 -1.0])

# C = Finch.Tensor(CSCFormat(), [1.0 0; 0 1.0])

# D = A * C

# E = Finch.Tensor(CSCFormat(), D)

# F = Sparse(E)

# println(A)
# println(typeof(A))
# println(finch_isapprox(A, B; rtol=1e-8, atol=1e-12))
# println(finch_isapprox(A, D; rtol=1e-8, atol=1e-12))
# println(finch_isapprox(A, E; rtol=1e-8, atol=1e-12))

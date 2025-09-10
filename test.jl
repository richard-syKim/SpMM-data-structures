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
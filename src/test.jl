using Pkg
Pkg.add("StatsBase")
Pkg.add("Finch")

using StatsBase
using Finch

A = [1 0; 1 0]
B = [0 1; 0 2]
D = [0 0 1; 0 2 0]
E = [1 0; 0 2; 0 0]
F_ = [1 0; 0 0; 0 2]
println(B)
A = Finch.Tensor(Finch.CSCFormat(), A)
B = Finch.Tensor(Finch.CSCFormat(), B)
D = Finch.Tensor(Finch.CSCFormat(), D)
E = Finch.Tensor(Finch.CSCFormat(), E)
F = Finch.Tensor(Finch.CSCFormat(), F_)
G = Finch.Tensor(Finch.CSFFormat(2), F_)
C = A * B
println(B)
println(D)
println(E)
println(F)
println(G)
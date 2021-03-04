using LinearAlgebra
using TSVD

# Basic Linear Algebra
A = [[1 2 3; 4 1 6; 7 8 1]; [1 2 3; 1 2 3; 1 2 3]]

B = rand(4,4,4)
Btilda = reshape(B,4,4^2)

# MPS
rank=4
k=2

A = rand((2 for _=1:rank)...)
sites = []

next = []
A_new = reshape(A, 2^(rank-1), 2^1)
next, s, V = TSVD.tsvd(A_new, k)
push!(sites, V)

A_new = reshape(next, 2^(rank-2), 2^2)
next, s, V = TSVD.tsvd(A_new, k)
push!(sites, reshape(V, 2, 2, 2))

A_new = reshape(next, 2^(rank-3), 2^2)
next, s, V = TSVD.tsvd(A_new, k)
push!(sites, reshape(V, 2, 2, 2))

# etc...

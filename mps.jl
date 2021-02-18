# A basic MPS calculation
using LinearAlgebra
using TSVD

D=4
k=2

A = rand((2 for _=1:D)...)
sites = []

next = []
A_new = reshape(A, 2^(D-1), 2^1)
next, s, V = TSVD.tsvd(A_new, k)
push!(sites, V)

A_new = reshape(next, 2^(D-2), 2^2)
next, s, V = TSVD.tsvd(A_new, k)
push!(sites, reshape(V, 2, 2, 2))

A_new = reshape(next, 2^(D-3), 2^2)
next, s, V = TSVD.tsvd(A_new, k)
push!(sites, reshape(V, 2, 2, 2))

print(sites)

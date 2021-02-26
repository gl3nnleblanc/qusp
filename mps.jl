# A basic MPS calculation
using LinearAlgebra
using TSVD

function mps(A, bond_dim=2)
    # Returns a matrix product state representation of A.
    # TODO: wrapper class to replace SITES supporting contraction with other tensor
    sites = []
    next = []
    rank = length(size(A))
    if rank<2
        return A
    end
    A_new = reshape(A, 2^(rank-1), 2^1)
    next, s, V = TSVD.tsvd(A_new, bond_dim)
    next = next * diagm(s)
    push!(sites, V)
    for i=2:rank-2
        print(i)
        A_new = reshape(next, 2^(rank-i), 2^2)
        next, s, V = TSVD.tsvd(A_new, bond_dim)
        next = next * diagm(s)
        push!(sites, reshape(V, 2, 2, 2))
    end
    A_new = reshape(next, 2, 2^2)
    next, s, V = TSVD.tsvd(A_new, bond_dim)
    push!(sites, V)
    #push!(next * diagm(s))
    return sites
 end

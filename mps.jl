module MatrixProductState

export mps
# A basic MPS calculation
using LinearAlgebra
using TSVD

"""
    truncate(s, dim)

Zero entries of `s` on indices greater than `dim`.
"""
function truncate(s, dim)
    for (i,_)=enumerate(s)
        if i > dim
            s[i]=0
        end
    end
    return s
end


"""
    mps(A, bond_dim)

Generate a matrix product state representation of the tensor `A`, using bond
dimension `bond_dim`. Limited to tensors having equal dimension on each index.
"""
function mps(A, bond_dim=2)
    for axis_dimension in size(A)
        if axis_dimension != size(A)[1]
            throw(DomainError(size(A), "mps is not supported for tensors hav" *
                              "ing nonuniform dimensions"))
        end
    end
    # TODO: wrapper class to replace SITES supporting contraction with other
    #   tensor
    # TODO: support choosing left/right canonical form
    sites = []
    next = []
    rank = length(size(A))
    if rank<2
        return A
    end
    A_new = reshape(A, 2^(rank-1), 2^1)
    next, s, V = svd(A_new)
    s = truncate(s, bond_dim)
    next = next * diagm(s)
    push!(sites, transpose(V))
    for i=2:rank-2
        print(i)
        A_new = reshape(next, 2^(rank-i), 2^2)
        next, s, V = svd(A_new)
        s = truncate(s, bond_dim)
        next = next * diagm(s)
        push!(sites, reshape(transpose(V), 2, 2, 2))
    end
    A_new = reshape(next, 2, 2^2)
    next, s, V = svd(A_new)
    s = truncate(s, bond_dim)
    push!(sites, transpose(V))
    push!(sites, next * diagm(s))
    return sites
 end

 end # module

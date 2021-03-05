module MatrixProductState

export mps
# A basic MPS calculation
using LinearAlgebra


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
    original_norm = norm(s)
    s = s[1:(length(s)<bond_dim ? end : bond_dim)]
    s *= original_norm / norm(s)
    next = next * diagm(s)
    push!(sites, transpose(V))
    next_axis_dim=length(s)
    for i=2:rank-2
        A_new = reshape(next, 2^(rank-i), next_axis_dim*2)
        next, s, V = svd(A_new)
        original_norm = norm(s)
        s = s[1:(length(s)<bond_dim ? end : bond_dim)]
        s *= original_norm / norm(s)
        V = transpose(V)
        V = V[1:(size(V)[1]<bond_dim ? end : bond_dim),:]
        next = next[:,1:(size(next)[2]<bond_dim ? end : bond_dim)] * diagm(s)
        next_axis_dim = 0
        if length(V) < bond_dim^2 * 2
            prev_axis_dim = size(sites[i-1])[1]
            next_axis_dim = length(V) ÷ (2 * prev_axis_dim)
            push!(sites, reshape(V, next_axis_dim, 2, prev_axis_dim))
        else
            push!(sites, reshape(V, bond_dim, 2, bond_dim))
            next_axis_dim = bond_dim
        end
    end
    if bond_dim > 3
        A_new = reshape(next, 2, 2^(next_axis_dim-1))
    else
        A_new = reshape(next, 2, 4)
    end
    next, s, V = svd(A_new)
    V = transpose(V)
    original_norm = norm(s)
    s = s[1:(length(s)<bond_dim ? end : bond_dim)]
    s *= original_norm / norm(s)
    push!(sites, reshape(V, 2, 2, length(V) ÷ 4))
    push!(sites, next * diagm(s))
    return sites
 end

 end # module

module MatrixProductState

export MPS, mps, contract_mps
# A basic MPS calculation
using LinearAlgebra


"""
    A wrapper type for a matrix product state.
"""
mutable struct MPS
    # TODO: support contraction with another tensor
    sites::Array{Array{Any}}
end


"""
    Contracts the MPS into a single tensor.
"""
function contract_mps(m::MPS)
    sites = m.sites
    rank = length(sites)
    axis_dim = div(length(sites[2]), 2)
    intermediate = reshape(sites[2], axis_dim, 2) * sites[1]
    for i = 2:rank-2
        axis_dim = size(sites[i+1])[3]
        left_axis_dim = div(length(sites[i+1]), axis_dim)
        right_axis_dim = div(length(intermediate), axis_dim)
        intermediate =
            reshape(sites[i+1], left_axis_dim, axis_dim) *
            reshape(intermediate, axis_dim, right_axis_dim)
    end
    res = sites[rank] * reshape(intermediate, 2, 2^(rank - 1))
    return reshape(res, (2 for _ = 1:rank)...)
end


"""
    truncate_and_renormalize(s, bond_dim)

Truncate vector of singular values and renormalize.
"""
function truncate_and_renormalize(s, bond_dim)
    original_norm = sqrt(dot(s, conj(s)))
    s = s[1:(length(s) < bond_dim ? end : bond_dim)]
    s *= original_norm / sqrt(dot(s, conj(s)))
    return s
end


"""
    mps(A, bond_dim)

Generate a matrix product state representation of the tensor `A`, using bond
dimension `bond_dim`. Limited to tensors having equal dimension on each index.
"""
function mps(A, bond_dim = 2)
    # TODO: support choosing left / right canonical form
    for axis_dimension in size(A)
        if axis_dimension != size(A)[1]
            throw(
                DomainError(
                    size(A),
                    "mps is not supported for tensors having nonuniform dimensions",
                ),
            )
        end
    end
    sites = []
    next = []
    rank::Int = length(size(A))
    if rank < 2
        return A
    end
    A_new = reshape(A, 2^(rank - 1), 2^1)
    next, s, V = svd(A_new)
    s = truncate_and_renormalize(s, bond_dim)
    next = next * diagm(s)
    push!(sites, conj(transpose(V)))
    next_axis_dim = length(s)
    for i = 2:rank-2
        A_new = reshape(next, 2^(rank - i), next_axis_dim * 2)
        next, s, V = svd(A_new)
        s = truncate_and_renormalize(s, bond_dim)
        V = conj(transpose(V))
        V = V[1:(size(V)[1] < bond_dim ? end : bond_dim), :]
        next = next[:, 1:(size(next)[2] < bond_dim ? end : bond_dim)] * diagm(s)
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
        A_new = reshape(next, 2, 2^(next_axis_dim - 1))
    else
        A_new = reshape(next, 2, 4)
    end
    next, s, V = svd(A_new)
    V = conj(transpose(V))
    s = truncate_and_renormalize(s, bond_dim)
    push!(sites, reshape(V, 2, 2, length(V) ÷ 4))
    push!(sites, next * diagm(s))
    return MPS(sites)
end

end # module

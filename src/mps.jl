module MatrixProductState

export MPS, mps, contract_mps, split_tensor
# A basic MPS calculation
using LinearAlgebra
using Einsum

"""
    A wrapper type for a matrix product state.
"""
mutable struct MPS
    # TODO: support contraction with another tensor
    sites::Array{Array{<:Number}}
    bond_dim::Integer
end


"""
    Contracts the MPS into a single tensor.
"""
function contract_mps(m::MPS)
    sites = m.sites
    rank = length(sites)

    right_edge = sites[1]
    next = sites[2]
    @einsum block[left_bond, q2, q1] := next[left_bond, q2, χ] * right_edge[χ, q1]
    for i = 3:rank-1
        bond_dim = size(block)[1]
        prev = reshape(block, bond_dim, :)
        next = sites[i]
        @einsum block[left_bond, qi, qs] := next[left_bond, qi, χ] * prev[χ, qs]
    end
    bond_dim = size(block)[1]
    prev = reshape(block, bond_dim, :)
    next = sites[rank]
    @einsum block[qi, qs] := next[qi, χ] * prev[χ, qs]
    return reshape(block, (2 for _=1:rank)...)
end


"""
    truncate_and_renormalize(s, bond_dim)

Truncate vector of singular values and renormalize.
"""
function truncate_and_renormalize(s, bond_dim)
    original_norm = sqrt(dot(s, s))
    if original_norm == 0
        return s[1:min(length(s), bond_dim)]
    end
    s = s[1:min(length(s), bond_dim)]
    s *= original_norm / sqrt(dot(s, s))
    return s
end


"""
    split_tensor(A, left_axis_dim, right_axis_dim, bond_dim)

Splits tensor `A` along leftmost axis with specified dimensions. Truncates
up to `bond_dim` singular values. Visually,

            `A` |-> -`next`-`site`-

where the rightmost dangling edge has dimension `right_axis_dim` and the
leftmost dangingle edge has dimension `left_axis_dim`. The virtual bond has
dimension `bond_dim`.

Returns outgoing edge dimension, left matrix, right matrix.
"""
function split_tensor(
    A::Array{<:Number},
    left_axis_dim::Integer,
    right_axis_dim::Integer,
    bond_dim::Integer,
)
    A_new = reshape(A, left_axis_dim, right_axis_dim)
    next, s, V_dag = svd(A_new)
    V = conj(transpose(V_dag))
    V = V[1:min(size(V)[1], bond_dim), :]
    s = truncate_and_renormalize(s, bond_dim)
    next = next[:, 1:min(size(next)[2], bond_dim)] * diagm(s)
    return length(s), next, V
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
    next_axis_dim, next, site = split_tensor(A, 2^(rank - 1), 2^1, bond_dim)
    push!(sites, site)
    for i = 2:rank-2
        next_axis_dim, next, V =
            split_tensor(next, 2^(rank - i), next_axis_dim * 2, bond_dim)
        if length(V) == bond_dim^2 * 2
            push!(sites, reshape(V, bond_dim, 2, bond_dim))
        else
            prev_axis_dim = length(V) ÷ (2 * next_axis_dim)
            push!(sites, reshape(V, next_axis_dim, 2, prev_axis_dim))
        end
    end
    prev_axis_dim = 0
    if bond_dim > 3
        prev_axis_dim = 2^(next_axis_dim - 1)
    else
        prev_axis_dim = 4
    end
    _, next, V = split_tensor(next, 2, prev_axis_dim, bond_dim)
    push!(sites, reshape(V, 2, 2, length(V) ÷ 4))
    push!(sites, next)
    return MPS(sites, bond_dim)
end

end # module

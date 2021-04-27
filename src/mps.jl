module MatrixProductState

export MPS, mps, contract_mps, split_tensor, set_orthogonality
# An MPS wrapper and associated functions
using LinearAlgebra
using Einsum


"""
    A wrapper type for a matrix product state.
"""
mutable struct MPS
    # TODO: support contraction with another tensor
    sites::Array{Array{<:Number}}
    bond_dim::Integer
    orthogonality_site::Integer
end


"""
    Moves the orthogonality center.
"""
function set_orthogonality(m::MPS, site::Integer)
    sites = copy(m.sites)
    curr_orthogonality_site = m.orthogonality_site
    N = length(m.sites)
    if !(site in 1:N)
        throw(DomainError(site, "Choose a site within bounds 1 to " * string(N)))
    end

    if site == curr_orthogonality_site
        return
    end

    if site > curr_orthogonality_site
        # Move O.C. from right to left
        # i = curr, curr + 1, curr + 2, etc. <--
        for i = curr_orthogonality_site:site
            active_site = sites[i]
            left_edge_dim = first(size(active_site))
            active_size = size(active_site)
            L, Q = lq(reshape(active_site, left_edge_dim, :))
            if first(size(L)) < first(size(Q))
                Q = Q[1:first(size(L)), :]
            end
            sites[i] = convert(Array{Float64}, reshape(Q, active_size...))
            if i < N
                next = sites[i+1]
                @einsum updated[χ_l, q, χ_r] := next[χ_l, q, χ] * L[χ, χ_r]
                sites[i+1] = updated
            else
                # Do nothing if at edge
                sites[i] = active_site
            end
        end
    else
        # Move O.C. from left to right
        for i = curr_orthogonality_site:-1:site
            active_site = sites[i]
            right_edge_dim = last(size(active_site))
            active_size = size(active_site)
            Q, R = qr(reshape(active_site, :, right_edge_dim))
            if last(size(R)) < last(size(Q))
                Q = Q[:, 1:last(size(R))]
            end
            sites[i] = convert(Array{Float64}, reshape(Q, active_size...))
            if i > 1
                next = sites[i-1]
                @einsum updated[χ_l, q, χ_r] := R[χ_l, χ] * next[χ, q, χ_r]
                sites[i-1] = updated
            else
                sites[i] = active_site
            end
        end
    end
    return MPS(sites, m.bond_dim, site)
end


"""
    Contracts the MPS into a single tensor. Moves right to left along chain.
"""
function contract_mps(m::MPS)
    sites = m.sites
    rank = length(sites)

    right_edge = sites[1]
    next = sites[2]
    @einsum block[χ_out, q2, q1] := next[χ_out, q2, χ] * right_edge[χ, q1]
    for i = 3:rank-1
        bond_dim = size(block)[1]
        prev = reshape(block, bond_dim, :)
        next = sites[i]
        @einsum block[χ_out, qi, qs] := next[χ_out, qi, χ] * prev[χ, qs]
    end
    bond_dim = size(block)[1]
    prev = reshape(block, bond_dim, :)
    next = sites[rank]
    @einsum block[qi, qs] := next[qi, χ] * prev[χ, qs]
    return reshape(block, (2 for _ = 1:rank)...)
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
    return MPS(sites, bond_dim, rank)
end

end # module

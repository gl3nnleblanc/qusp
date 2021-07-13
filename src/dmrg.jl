module TimeEvolvingBlockDecimation

export Hamiltonian, block_evolve, tebd

include("./mps.jl")
using .MatrixProductState
using Einsum
using Zygote

"""
    A 1D hamiltonian with a local term and nearest neighbor interactions.
"""
mutable struct Hamiltonian
    interaction::Array{<:Number,2}
    field::Array{<:Number,2}
end


"""
    Applies a single block evolution step to ψ.
"""
function block_evolve(ψ::MPS, H::Hamiltonian, t::Number)
    N = length(ψ.sites)
    ψ = set_orthogonality(ψ, N)
    updated_sites::Array{Array{<:Number}} = []
    interaction = reshape(exp(H.interaction * t), 2, 2, 2, 2)
    field = exp(H.field * t)
    half_field = exp(H.field * t / 2)
    # Left to Right
    local previous_right_site::Array{<:Number}
    local evolved::Array{<:Number}

    leftmost_site = ψ.sites[N]
    second_leftmost_site = ψ.sites[N-1]
    right_edge_dim = size(second_leftmost_site)[3]
    @einsum block[q1, q2, right] :=
        leftmost_site[q1, chi] * second_leftmost_site[chi, q2, right]
    @einsum evolved[q1, q2, right] :=
        interaction[q1, q2, c, d] * field[c, a] * half_field[d, b] * block[a, b, right]
    new_right_dim = size(evolved)[3]
    _, new_left_site, new_right_site =
        split_tensor(evolved, 2, 2 * right_edge_dim, ψ.bond_dim)
    push!(updated_sites, new_left_site)
    previous_right_site = reshape(new_right_site, size(second_leftmost_site)...)

    for i = N-2:-1:2
        right_site = ψ.sites[i]
        left_site = previous_right_site
        right_edge_dim = size(right_site)[3]
        left_edge_dim = size(left_site)[1]
        @einsum block[left, q1, q2, right] :=
            left_site[left, q1, chi] * right_site[chi, q2, right]
        @einsum evolved[left, q1, q2, right] :=
            interaction[q1, q2, c, d] *
            half_field[c, a] *
            half_field[d, b] *
            block[left, a, b, right]
        new_bond_dim, new_left_site, new_right_site =
            split_tensor(evolved, left_edge_dim * 2, right_edge_dim * 2, ψ.bond_dim)
        new_left_edge_dim = size(left_site)[1]
        new_right_edge_dim = size(right_site)[3]
        push!(updated_sites, reshape(new_left_site, new_left_edge_dim, 2, new_bond_dim))
        previous_right_site = reshape(new_right_site, new_bond_dim, 2, new_right_edge_dim)
    end
    rightmost_site = ψ.sites[1]
    left_site = previous_right_site
    left_edge_dim = size(left_site)[1]
    @einsum block[left, q1, q2] := left_site[left, q1, chi] * rightmost_site[chi, q2]
    @einsum evolved[left, q1, q2] :=
        interaction[q1, q2, c, d] * half_field[c, a] * field[d, b] * block[left, a, b]
    _, new_left_site, new_right_site =
        split_tensor(evolved, 2 * left_edge_dim, 2, ψ.bond_dim)
    push!(updated_sites, reshape(new_left_site, size(left_site)...))
    push!(updated_sites, reshape(new_right_site, size(rightmost_site)...))
    res = MPS(reverse(updated_sites), ψ.bond_dim, 1)
    return set_orthogonality(res, N)
end


"""
    Performs time evolving block decimation on ψ. As `num_steps` increases
    the fidelity against the true time-evolved state increases (per
    Suzuki-Trotter decomposition).
"""
function tebd(ψ::MPS, H::Hamiltonian, angle::Number, resolution::Integer = 1)
    ψ_res = ψ
    for _ = 1:resolution
        ψ_res = block_evolve(ψ_res, H, angle / resolution)
    end
    return ψ_res
end

end # module

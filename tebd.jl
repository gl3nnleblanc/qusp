module TimeEvolvingBlockDecimation

export Hamiltonian, block_evolve

include("./mps.jl")
using .MatrixProductState
using Einsum

"""
    A 1D hamiltonian with a local term and nearest neighbor interactions.
"""
mutable struct Hamiltonian
    interaction::Array{<:Number,2}
    field::Array{<:Number,2}
end


"""
    Applies eᴴᵗ to ψ. 
    Performs block evolution from left to right.
"""
function block_evolve(ψ::MPS, H::Hamiltonian, t::Number)
    updated_sites::Array{Array{<:Number}} = []
    interaction = reshape(exp(H.interaction * t), 2, 2, 2, 2)
    field = exp(H.field * t)
    half_field = exp(H.field * t / 2)
    N = length(ψ.sites)
    # Left to Right
    local previous_right_site::Array{<:Number}
    for i = N-1:-1:1
        right_site = ψ.sites[i]
        local evolved::Array{<:Number}

        if i == N - 1
            left_site = ψ.sites[i+1]
        else
            left_site = previous_right_site
        end

        if i == N - 1
            @einsum block[q1, q2, right] := left_site[q1, chi] * right_site[chi, q2, right]
            @einsum evolved[q1, q2, right] :=
                interaction[q1, q2, d, c] *
                field[c, a] *
                half_field[d, b] *
                block[a, b, right]
            _, new_left_site, new_right_site =
                split_tensor(evolved, 2, 2 * ψ.bond_dim, ψ.bond_dim)
            push!(updated_sites, new_left_site)
            previous_right_site = reshape(new_right_site, size(right_site)...)

        elseif i > 1
            @einsum block[left, q1, q2, right] :=
                left_site[left, q1, chi] * right_site[chi, q2, right]
            @einsum evolved[left, q1, q2, right] :=
                interaction[q1, q2, d, c] *
                half_field[c, a] *
                half_field[d, b] *
                block[left, a, b, right]
            _, new_left_site, new_right_site =
                split_tensor(evolved, 2 * ψ.bond_dim, 2 * ψ.bond_dim, ψ.bond_dim)
            push!(
                updated_sites,
                reshape(
                    new_left_site,
                    length(new_right_site) ÷ (2 * ψ.bond_dim),
                    2,
                    ψ.bond_dim,
                ),
            )
            previous_right_site = reshape(new_right_site, size(right_site)...)

        elseif i == 1
            @einsum block[left, q1, q2] := left_site[left, q1, chi] * right_site[chi, q2]
            @einsum evolved[left, q1, q2] :=
                interaction[q1, q2, d, c] *
                half_field[c, a] *
                field[d, b] *
                block[left, a, b]
            _, new_left_site, new_right_site =
                split_tensor(evolved, 2 * ψ.bond_dim, 2, ψ.bond_dim)
            push!(updated_sites, reshape(new_left_site, size(left_site)...))
            push!(updated_sites, reshape(new_right_site, size(right_site)...))
        end
    end
    return MPS(reverse(updated_sites), ψ.bond_dim)
end

end # module

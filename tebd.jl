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
    G = zeros(2,2,2,2)
    G[1, 1, 1, 1] = 1
    G[2, 2, 1, 1] = 1
    G[2, 2, 2, 2] = 1
    G[1, 1, 2, 2] = 1
    #@einsum res[q1, q2] := G[a, q1, c, q2] * ψ1[a] * ψ2[c]
    N = length(ψ.sites)
    # left to right
    for i = length(ψ.sites):-1:2
        left_site = ψ.sites[i]
        local evolved::Array{<:Number}
        local left_site::Array{<:Number}

        if i == length(ψ.sites)
            right_site = ψ.sites[i-1]
        else
            right_site = updated_sites[i-1]
        end

        # Left edge
        if i == N
            @einsum block[left, q1, q2] := left_site[left, q1, chi] * right_site[chi, q2]
            @einsum evolved[left, q1, q2] := block[left, a, b] * G[a, q1, b, q2]
            _, new_left_site, new_right_site =
                split_tensor(evolved, 2 * ψ.bond_dim, 2, ψ.bond_dim)
            push!(updated_sites, reshape(
                new_left_site,
                ψ.bond_dim,
                2,
                length(new_left_site) ÷ (2 * ψ.bond_dim),
               ))
            push!(updated_sites, reshape(new_right_site, length(new_right_site) ÷ 2, 2))

            push!(updated_sites, new_left_site)
            # Middle pairs
        elseif i < N
            @einsum block[left, q1, q2, right] :=
                left_site[left, q1, chi] * right_site[chi, q2, right]
            @einsum evolved[left, q1, q2, right] := block[left, a, b, right] * G[a, q1, b, q2]
            _, new_left_site, new_right_site =
                split_tensor(evolved, 2 * ψ.bond_dim, 2 * ψ.bond_dim, ψ.bond_dim)
            updated_sites[i-1] = reshape(new_left_site, ψ.bond_dim, 2, ψ.bond_dim)
            push!(
                updated_sites,
                reshape(
                    new_right_site,
                    length(new_right_site) ÷ (2 * ψ.bond_dim),
                    2,
                    ψ.bond_dim,
                ),
            )
        # Right edge
        elseif i == 2
            @einsum block[q1, q2, right] := left_site[q1, chi] * right_site[chi, q2, right]
            @einsum evolved[q1, q2, right] := block[a, b, right] * G[a, q1, b, q2]
            _, new_left_site, new_right_site =
                split_tensor(evolved, 2, 2 * ψ.bond_dim, ψ.bond_dim)
            push!(updated_sites, new_left_site)
            push!(
                updated_sites,
                reshape(
                    new_right_site,
                    length(new_right_site) ÷ (2 * ψ.bond_dim),
                    2,
                    ψ.bond_dim,
                ),
            )
        end
    end
    return MPS(updated_sites, ψ.bond_dim)
end

end # module

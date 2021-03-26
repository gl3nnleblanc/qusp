module TimeEvolvingBlockDecimation

export Hamiltonian, block_evolve

include("./mps.jl")
using .MatrixProductState
using TensorOperations

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
    N = length(ψ.sites)
    for i = 2:length(ψ.sites)
        right_site = ψ.sites[i]
        local evolved::Array{<:Number}
        local left_site::Array{<:Number}

        if i == 2
            left_site = ψ.sites[i-1]
        else
            left_site = updated_sites[i-1]
        end

        if i == 2
            @tensor begin
                block[q1, q2, right] := left_site[q1, chi] * right_site[chi, q2, right]
                evolved[q1, q2, right] :=
                    block[a, b, right] *
                    interaction[a, b, c, d] *
                    field[c, q1] *
                    field[d, q2]
            end
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
        elseif i < N
            @tensor begin
                block[left, q1, q2, right] :=
                    left_site[left, q1, chi] * right_site[chi, q2, right]
                evolved[left, q1, q2, right] :=
                    block[left, a, b, right] *
                    interaction[a, b, c, d] *
                    field[c, q1] *
                    field[d, q2]
            end
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
        elseif i == N
            @tensor begin
                block[left, q1, q2] := left_site[left, q1, chi] * right_site[chi, q2]
                evolved[left, q1, q2] :=
                    block[left, a, b] *
                    interaction[a, b, c, d] *
                    field[c, q1] *
                    field[d, q2]
            end
            _, new_left_site, new_right_site =
                split_tensor(evolved, 2 * ψ.bond_dim, 2, ψ.bond_dim)
            updated_sites[i-1] = reshape(
                new_left_site,
                ψ.bond_dim,
                2,
                length(new_left_site) ÷ (2 * ψ.bond_dim),
            )
            push!(updated_sites, reshape(new_right_site, length(new_right_site) ÷ 2, 2))
        end
    end
    return MPS(updated_sites, ψ.bond_dim)
end

end # module

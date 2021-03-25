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
"""
function block_evolve(ψ::MPS, H::Hamiltonian, t::Number)
    interaction = exp(H.interaction * t)
    field = exp(H.field * t)
    N = length(ψ.sites)
    for (i, site) in enumerate(ψ.sites)
        if i == 1
            neighbor = ψ.sites[i+1]
            @tensor block[q1, q2, right] := site[q1, chi] * neighbor[chi, q2, right]
        elseif i < N - 1
            neighbor = ψ.sites[i+1]
            @tensor block[left, q1, q2, right] :=
                site[left, q1, chi] * neighbor[chi, q2, right]
        elseif i == N - 1
            neighbor = ψ.sites[i+1]
            @tensor block[left, q1, q2] := site[left, q1, chi] * neighbor[chi, q2]
        end
    end
    return 0
end

end # module

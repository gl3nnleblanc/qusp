module TimeEvolvingBlockDecimation

export Hamiltonian, block_evolve

include("./mps.jl")
using .MatrixProductState

"""
    A 1D hamiltonian with a local term and nearest neighbor interactions.
"""
struct Hamiltonian
    interaction::Array
    field::Array
end


"""
    Applies eᴴᵗ to ψ. 
"""
function block_evolve(ψ::MPS, H::Hamiltonian, t::Number)
    H.interaction *= t
    H.field *= t
    print(H.field)
    N = length(ψ.sites)
    for (i, site) in enumerate(ψ.sites)
        if i < N
            neighbor = ψ.sites[i+1]
        end
    end
    return 0
end

end # module

using LinearAlgebra
using Test

include("./tebd.jl")
using .TimeEvolvingBlockDecimation
using .TimeEvolvingBlockDecimation.MatrixProductState

@testset "Time Evolving Block Decimation Tests" begin
    ⊗ = kron
    σ_x = [
        0 1
        1 0
    ]
    σ_z = [
        1 0
        0 -1
    ]
    ising = Hamiltonian(σ_z ⊗ σ_z, σ_x)
    function ising_matrix(sites::Integer)
        I = [
            1 0
            0 1
        ]
        identity_string = [I for _ = 1:sites]
        local_term = zeros(2^sites, 2^sites)
        interaction_term = zeros(2^sites, 2^sites)

        for i = 1:sites
            pauli_string = copy(identity_string)
            pauli_string[i] = σ_x
            local_term += reduce(kron, pauli_string)
        end

        for i = 1:sites-1
            pauli_string = copy(identity_string)
            pauli_string[i] = σ_z
            pauli_string[i+1] = σ_z
            interaction_term += reduce(kron, pauli_string)
        end
        return interaction_term + local_term
    end
    @testset "Block Decimation" begin
        sites = 3
        ψ = ones(ComplexF32, (2 for _ = 1:sites)...)
        ψ /= sqrt(dot(ψ, ψ))
        ψ_mps = mps(ψ)
        ψ_res = reshape(contract_mps(block_evolve(ψ_mps, ising, 0)), 2^sites)

        ϕ = ones(ComplexF32, 2^sites)
        ϕ /= sqrt(dot(ϕ, ϕ))
        I = [1 0; 0 1]
        H = ising_matrix(sites)
        ϕ_res = exp(H * 0) * ϕ

        @assert dot(ψ_res, ϕ_res) == 1

    end
end

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
    @testset "Error Handling" begin end
    @testset "Hamiltonians" begin end
    @testset "Block Decimation" begin
        ψ = rand(ComplexF32, (2 for _ = 1:6)...)
        ψ /= sqrt(dot(ψ, ψ))
        ψ_mps = mps(ψ)
        block_evolve(ψ_mps, ising, 2)
    end
end

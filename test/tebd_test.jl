using LinearAlgebra
using Einsum
using Test

include("../src/QuSP.jl")
using Main.QuSP.TimeEvolvingBlockDecimation
using Main.QuSP.MatrixProductState


@testset "Einsum Tests" begin
    ψ1 = [0; 1]
    ψ2 = [0; 1]
    ⊗ = kron
    ψ = ψ1 ⊗ ψ2
    X = [
        0 1
        1 0
    ]
    Y = [
        0 -1
        1 0
    ]
    G = reshape(X ⊗ Y, 2, 2, 2, 2)

    # Correct index order for 2-qubit block contraction
    @einsum res[q1, q2] := G[q1, q2, b, a] * ψ1[a] * ψ2[b]
    @test reshape(res, 4) == (X ⊗ Y) * ψ
    # Degree of freedom in index order
    @einsum res[q1, q2] := G[q1, q2, a, b] * ψ1[a] * ψ2[b]
    @test reshape(res, 4) == (X ⊗ Y) * ψ
    # Incorrect order
    @einsum res[q1, q2] := G[a, b, q1, q2] * ψ1[a] * ψ2[b]
    @test reshape(res, 4) != (X ⊗ Y) * ψ

    # Fixed random local contraction on two sites
    A = [
        0.293865 0.887788
        0.191751 0.2021
    ]
    B = [
        0.835162 0.797031
        0.292658 0.0414362
    ]
    ψ_rand1 = [
        0.0497384
        0.8757355
    ]
    ψ_rand2 = [
        0.6306069
        0.360918
    ]
    ψ = ψ_rand1 ⊗ ψ_rand2
    G = A ⊗ B
    res_m = G * ψ
    G = reshape(G, 2, 2, 2, 2)
    @einsum res_t[q1, q2] := G[q1, q2, b, a] * ψ_rand1[a] * ψ_rand2[b]
    for e in (res_m - reshape(res_t, 4))
        @test e < 1e-10
    end

    # Fixed random block + local contraction on two sites:
    #     ψ = C * (A ⊗ B) * ψ_0 (no entanglement originally)
    A = [
        0.443606 0.629362
        0.259218 0.650738
    ]
    B = [
        0.855656 0.447434
        0.170226 0.680784
    ]
    C = [
        0.60382 0.698733 0.242626 0.473168
        0.683629 0.141109 0.327527 0.183103
        0.956508 0.0732721 0.741199 0.780935
        0.842576 0.616262 0.217417 0.848626
    ]
    ψ = ψ_rand1 ⊗ ψ_rand2
    res_m = C * (A ⊗ B) * ψ
    C_t = reshape(C, 2, 2, 2, 2)
    @einsum res_t[q1, q2] := C_t[q1, q2, d, c] * B[d, b] * A[c, a] * ψ_rand1[a] * ψ_rand2[b]
    for e in (res_m - reshape(res_t, 4))
        @test e < 1e-10
    end

    # Fixed random block + local contraction on two sites:
    #     ψ = C * (A ⊗ B) * ψ_0 (entangled start)
    ψ = [0.692; 0.6442; 0.169; 0.873]
    res_m = C * (A ⊗ B) * ψ
    ψ = reshape(ψ, 2, 2)
    C_t = reshape(C, 2, 2, 2, 2)
    @einsum res_t[q1, q2] := C_t[q1, q2, d, c] * B[d, b] * A[c, a] * ψ[a, b]
    for e in (res_m - reshape(res_t, 4))
        @test e < 1e-10
    end

    I = zeros(2, 2, 2, 2)
    I[1, 1, 1, 1] = 1
    I[1, 1, 2, 2] = 1
    I[2, 2, 1, 1] = 1
    I[2, 2, 2, 2] = 1
    @einsum res[q1, q2] := I[a, q1, c, q2] * ψ1[a] * ψ2[c]
    @test reshape(res, 4) == [0, 0, 0, 1]

    G = zeros(2, 2, 2, 2)
    G[1, 2, 1, 1] = 1
    G[2, 1, 1, 1] = 1
    G[1, 2, 2, 2] = 1
    G[2, 1, 2, 2] = 1
    @einsum res[q1, q2] := G[a, q1, c, q2] * ψ1[a] * ψ2[c]
    @test reshape(transpose(res), 4) == [0, 1, 0, 0]
end

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
    function ising_matrix(sites::Integer, local_only::Bool = false)
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
        if local_only
            return local_term
        end
        return interaction_term + local_term
    end
    @testset "Block Decimation" begin
        # Apply identity to |00...01⟩
        sites = 10
        ψ = zeros(ComplexF32, (2 for _ = 1:sites)...)
        ψ[(2 for _ = 1:sites)...] = 1

        ψ /= sqrt(dot(ψ, ψ))
        ψ_mps = mps(ψ)
        ψ_res = reshape(contract_mps(block_evolve(ψ_mps, ising, 0)), 2^sites)

        ϕ = zeros(ComplexF32, 2^sites)
        ϕ[2^sites] = 1
        ϕ /= sqrt(dot(ϕ, ϕ))
        I = [1 0; 0 1]
        H = ising_matrix(sites)
        ϕ_res = exp(H * 0) * ϕ

        @test dot(ψ_res, ϕ_res) == 1

        # Apply only local interactions to |00...01⟩
        function do_local_spin_test(angle)
            ising = Hamiltonian(0 * (σ_z ⊗ σ_z), σ_x)
            sites = 5
            ψ = zeros(ComplexF32, (2 for _ = 1:sites)...)
            ψ[(2 for _ = 1:sites)...] = 1

            ψ /= sqrt(dot(ψ, ψ))
            ψ_mps = mps(ψ)
            ψ_res = reshape(contract_mps(block_evolve(ψ_mps, ising, angle * 1im)), 2^sites)

            ϕ = zeros(ComplexF32, 2^sites)
            ϕ[2^sites] = 1
            ϕ /= sqrt(dot(ϕ, ϕ))
            H = ising_matrix(sites, true)
            ϕ_res = exp(H * angle * 1im) * ϕ
            return abs(abs(dot(ψ_res, ϕ_res)) - 1) < 1e-10
        end
        angles = [π / 17, π / 6, π / 3, π / 2, π, 2 * π]
        @test all(map(do_local_spin_test, angles))

        # Fixed random test with interaction term
        ψ = [
            0.2007953f0 + 0.75265944f0im
            0.80030394f0 + 0.41267908f0im
            0.5720874f0 + 0.7258271f0im
            0.24059021f0 + 0.2459718f0im
            0.30385184f0 + 0.5990107f0im
            0.13170183f0 + 0.80890477f0im
            0.02885878f0 + 0.19243884f0im
            0.49203765f0 + 0.5239831f0im
            0.84893835f0 + 0.8569777f0im
            0.031204581f0 + 0.23515308f0im
            0.062227964f0 + 0.6164305f0im
            0.6660845f0 + 0.24422407f0im
            0.91392064f0 + 0.63511574f0im
            0.44659543f0 + 0.34430194f0im
            0.8511541f0 + 0.49638438f0im
            0.33232164f0 + 0.54759526f0im
        ]
        ψ /= norm(ψ)
        ψ_t = reshape(ψ, 2, 2, 2, 2)
        ψ_mps = mps(ψ_t, 4)
        ising = Hamiltonian(σ_z ⊗ σ_z, σ_x)
        H = ising_matrix(4)

        angle = π / 6 * -1im
        ψ_res = reshape(contract_mps(tebd(ψ_mps, ising, angle, 100)), 2^4)
        ϕ_res = exp(H * angle) * ψ
        @test abs(dot(ψ_res, ϕ_res) - 1) < 0.003
    end
end

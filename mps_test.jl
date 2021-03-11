using LinearAlgebra
using Test

include("./mps.jl")
using .MatrixProductState

@testset "Matrix Product State Tests" begin
    @testset "Error Handling" begin
        A = rand(1, 2)
        err = nothing
        try
            mps(A)
        catch err
        end
        @test err isa DomainError
    end
    @testset "Contraction" begin
        # First simple test
        A = ones(2, 2, 2)
        A[1, 1, 1] = 0
        A[1, 2, 2] = 0
        A[2, 1, 2] = 0
        A[2, 2, 1] = 0
        A[1, 1, 2] = 2
        A_mps = contract_mps(mps(A))
        @test abs(dot(A, A_mps) / dot(A, A) - 1) < 1e-7

        # Second simple test
        A = zeros(2, 2, 2)
        A[1, 1, 1] = -1
        A[1, 2, 2] = 1
        A[2, 1, 2] = -1
        A[2, 2, 1] = 1
        sites = mps(A)
        A_mps = contract_mps(mps(A))
        @test abs(dot(A, A_mps) / dot(A, A) - 1) < 1e-7

        # Bigger test
        rank = 4
        #A = zeros((2 for _=1:rank)...)
        #A[(1 for _=1:rank)...] = 1
        #A[(2 for _=1:rank)...] = -1
        A = rand((2 for _ = 1:rank)...)
        A = A / sqrt(dot(A, A))
        A_mps = contract_mps(mps(A, 512))
        @test abs(dot(A_mps, A) - 1) < 1e-7

        # Complex values test
        rank = 10
        A = rand(ComplexF64, (2 for _ = 1:rank)...)
        A = A / sqrt(dot(A, conj(A)))
        A_mps = contract_mps(mps(A, 2^5))
        @test abs(dot(conj(A_mps), A)) - 1 < 1e-7
    end
end

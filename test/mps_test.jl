using Einsum
using LinearAlgebra
using Test

include("../src/mps.jl")
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
        A = zeros(2, 2, 2)
        A_mps = mps(A)
        err = nothing
        try
            A_mps = set_orthogonality(A_mps, 4)
        catch err
        end
        @test err isa DomainError
        A = reshape(
            [
                0.464654 0.594944
                0.0734789 0.514704
                0.118756 0.799922
                0.142388 0.622209
                0.0820784 0.245697
                0.143912 0.291454
                0.967784 0.382256
                0.273797 0.880877
            ],
            2,
            2,
            2,
            2,
        )
        A_mps = mps(A)
        B_mps = mps(A)
        for i = 1:3
            A_mps = set_orthogonality(A_mps, i)
            @test round.(contract_mps(A_mps), digits = 5) ==
                  round.(contract_mps(B_mps), digits = 5)
        end
    end
    @testset "Orthogonality Center" begin
        # Output should be entirely right normal except for last site
        A = ones(2, 2, 2)
        A[1, 1, 1] = 0
        A[1, 2, 2] = 0
        A[2, 1, 2] = 0
        A[2, 2, 1] = 0
        A[1, 1, 2] = 2
        A_mps = mps(A)
        middle_site = A_mps.sites[2]
        @einsum res[i, j] := middle_site[i, a, b] * middle_site[j, a, b]
        @test round.(res, digits = 10) == [1 0; 0 1]

        A = rand((2 for _ = 1:10)...)
        A_mps = mps(A)
        for i = 2:9
            site = A_mps.sites[i]
            @einsum res[i, j] := site[i, a, b] * site[j, a, b]
            @test round.(res, digits = 7) == [1 0; 0 1]
        end
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
        A = rand((2 for _ = 1:rank)...)
        A = A / sqrt(dot(A, A))
        A_mps = contract_mps(mps(A, 512))
        @test abs(dot(A_mps, A) - 1) < 1e-7

        # Complex values test
        rank = 10
        A = rand(ComplexF64, (2 for _ = 1:rank)...)
        A = A / sqrt(dot(A, A))
        A_mps = contract_mps(mps(A, 4))
        @test abs(dot(A, A)) - 1 < 1e-7
        @test abs(dot(A_mps, A_mps)) - 1 < 1e-7
    end
end

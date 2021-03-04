using LinearAlgebra
using Test

include("./mps.jl")
using .MatrixProductState

@testset "Matrix Product State Tests" begin
    @testset "Error Handling" begin
        A = rand(1, 2)
        err = nothing
        try mps(A)
            catch err
        end
        @test err isa DomainError
    end
    @testset "Contraction" begin
        # First simple test
        A = ones(2,2,2)
        A[1,1,1] = 0
        A[1,2,2] = 0
        A[2,1,2] = 0
        A[2,2,1] = 0
        A[1,1,2] = 2
        sites = mps(A)
        intermediate = reshape(sites[2], 4, 2) * sites[1]
        A_mps = reshape(sites[3] * reshape(intermediate, 2, 4), 2, 2, 2)
        @test abs(dot(A, A_mps) / dot(A, A) - 1) < 1e-7

        # Second simple test
        A = zeros(2,2,2)
        A[1,1,1] = -1
        A[1,2,2] = 1
        A[2,1,2] = -1
        A[2,2,1] = 1
        sites = mps(A)
        intermediate = reshape(sites[2], 4, 2) * sites[1]
        A_mps = reshape(sites[3] * reshape(intermediate, 2, 4), 2, 2, 2)
        @test abs(dot(A, A_mps) / dot(A, A) - 1) < 1e-7

        # Bigger test
        rank = 6
        A = zeros((2 for _=1:rank)...)
        A[(1 for _=1:rank)...] = 1
        A[(2 for _=1:rank)...] = -1
        sites = mps(A)
        intermediate = reshape(sites[2], 4, 2) * sites[1]
        for i=2:rank-2
            intermediate = reshape(sites[i+1], 4, 2) * reshape(intermediate, 2, 2^i)
        end
        A_mps = sites[rank] * reshape(intermediate, 2, 2^(rank-1))
        @test dot(A_mps, A) / dot(A, A) == 1
    end
end

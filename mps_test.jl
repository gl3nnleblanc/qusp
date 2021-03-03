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
        sites = mps(A)
        intermediate = transpose(reshape(sites[2], 2, 4)) * sites[3]
        intermediate = reshape(intermediate, 2, 2, 2)
        intermediate = sites[1] * reshape(intermediate, 2, 4)
        A_mps = reshape(intermediate, 2, 2, 2)
        @test abs(dot(A, A_mps) - 4) < 1e-6

        # Second simple test
        A = zeros(2,2,2)
        A[1,1,1] = -1
        A[1,2,2] = 1
        A[2,1,2] = -1
        A[2,2,1] = 1
        sites = mps(A)
        intermediate = transpose(reshape(sites[2], 2, 4)) * sites[3]
        intermediate = reshape(intermediate, 2, 2, 2)
        intermediate = sites[1] * reshape(intermediate, 2, 4)
        A_mps = reshape(intermediate, 2, 2, 2)
        print(A_mps)
        print("\n\n\n")
        print(A)
        @test abs(dot(A, A_mps) - 20) < 1e-6
    end
end

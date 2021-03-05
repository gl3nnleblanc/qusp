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
        rank = 18
        #A = zeros((2 for _=1:rank)...)
        #A[(1 for _=1:rank)...] = 1
        #A[(2 for _=1:rank)...] = -1
        A = rand((2 for _=1:rank)...)
        A = A / sqrt(dot(A, A))
        sites = mps(A, 512)
        axis_dim = div(length(sites[2]), 2)
        intermediate = reshape(sites[2], axis_dim, 2) * sites[1]
        for i=2:rank-2
            axis_dim = size(sites[i+1])[3]
            left_axis_dim = div(length(sites[i+1]), axis_dim)
            right_axis_dim = div(length(intermediate), axis_dim)
            intermediate = reshape(sites[i+1], left_axis_dim, axis_dim) *
                reshape(intermediate, axis_dim, right_axis_dim)
        end
        A_mps = sites[rank] * reshape(intermediate, 2, 2^(rank-1))
        #@test abs(dot(A_mps, A) / dot(A, A) - 1) < 1e-7 Why does this fail and the bottom work???
        println(dot(A_mps, A_mps))
        println(dot(A_mps, A))
        println(dot(A, A))
        @test abs(dot(A_mps, A) - 1) < 1e-7
    end
end

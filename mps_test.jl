using Test

include("./mps.jl")
using .MatrixProductState

@testset "Matrix Product State Tests" begin
    @testset "Error Handling" begin
        A = rand(2, 2, 2, 2, 4)
        err = nothing
        try mps(A)
            catch err
        end
        @test err isa DomainError
    end
    @testset "Contraction" begin
        A = rand((2 for _=1:10)...)
        sites = mps(A)
        # TODO: contract and test mps(A) == A
        @test A != mps(A)
    end
end

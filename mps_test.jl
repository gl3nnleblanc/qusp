using Test

include("./mps.jl")
using .MatrixProductState

@testset "Matrix Product State Tests" begin
    A = rand((2 for _=1:10)...)
    sites = mps(A)
    # TODO: contract and test mps(A) == A
    @test A != mps(A)
end

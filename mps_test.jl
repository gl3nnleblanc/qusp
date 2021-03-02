using Test

@testset "Matrix Product State Tests" begin
    A = rand((2 for _=1:10)...)
    sites = mps(A, bond_dim=5)
    # TODO: contract and test mps(A) == A
end

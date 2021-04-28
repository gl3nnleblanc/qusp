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
        # Moving the orthogonality center
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
        # A:
        # .-<-<-<
        # | | | |
        #
        # B:
        # >->->-.
        # | | | |
        B_mps = set_orthogonality(A_mps, 1)
        @test round.(contract_mps(A_mps), digits = 5) ==
              round.(contract_mps(B_mps), digits = 5)
        @test A_mps.sites != B_mps.sites
        # Ensure each left-normal site in B is actually left-normal
        #     Middle 3-leg tensors
        for i = 2:3
            site = B_mps.sites[i]
            @einsum res[i, j] := site[a, b, i] * site[a, b, j]
            @test round.(res, digits = 8) == [1 0; 0 1]
        end
        # Left edge 2-leg tensor
        res = B_mps.sites[4] * transpose(B_mps.sites[4])
        @test round.(res, digits = 8) == [1 0; 0 1]
        # A:
        # .-<-<-<
        # | | | |
        #
        # C:
        # >->-.-<
        # | | | |
        C_mps = set_orthogonality(A_mps, 2)
        @test round.(contract_mps(A_mps), digits = 5) ==
              round.(contract_mps(C_mps), digits = 5)
        @test A_mps.sites != C_mps.sites
        # Left and right edge 2-leg tensors
        for i in [1, 4]
            res = C_mps.sites[i] * transpose(C_mps.sites[i])
            @test round.(res, digits = 8) == [1 0; 0 1]
        end
        # Second from left left-normal 3-leg tensor
        site = C_mps.sites[3]
        @einsum res[i, j] := site[a, b, i] * site[a, b, j]
        @test round.(res, digits = 8) == [1 0; 0 1]
        # A:
        # .-<-<-<
        # | | | |
        #
        # D:
        # >-.-<-<
        # | | | |
        D_mps = set_orthogonality(A_mps, 3)
        @test round.(contract_mps(A_mps), digits = 5) ==
              round.(contract_mps(D_mps), digits = 5)
        @test A_mps.sites != D_mps.sites
        # Left and right edge 2-leg tensors
        for i in [1, 4]
            res = D_mps.sites[i] * transpose(D_mps.sites[i])
            @test round.(res, digits = 8) == [1 0; 0 1]
        end
        # Second from right right-normal 3-leg tensor
        site = D_mps.sites[2]
        @einsum res[i, j] := site[i, a, b] * site[j, a, b]
        @test round.(res, digits = 8) == [1 0; 0 1]
        # D:
        # >-.-<-<
        # | | | |
        #
        # E:
        # >->-.-<
        # | | | |
        E_mps = set_orthogonality(D_mps, 2)
        # Test numerical stability: safe up to approx. 12 digits
        @test round.(contract_mps(A_mps), digits = 12) ==
              round.(contract_mps(D_mps), digits = 12)
        @test round.(contract_mps(A_mps), digits = 12) ==
              round.(contract_mps(E_mps), digits = 12)
        @test A_mps.sites != E_mps.sites
        @test D_mps.sites != E_mps.sites
        # Left and right edge 2-leg tensors
        for i in [1, 4]
            res = E_mps.sites[i] * transpose(E_mps.sites[i])
            @test round.(res, digits = 8) == [1 0; 0 1]
        end
        # Second from left left-normal 3-leg tensor
        site = E_mps.sites[3]
        @einsum res[i, j] := site[a, b, i] * site[a, b, j]
        @test round.(res, digits = 8) == [1 0; 0 1]
        # D:
        # >-.-<-<
        # | | | |
        #
        # F:
        # .-<-<-<
        # | | | |
        F_mps = set_orthogonality(D_mps, 4)
        # Test numerical stability: safe up to approx. 12 digits
        @test round.(contract_mps(A_mps), digits = 12) ==
              round.(contract_mps(D_mps), digits = 12)
        @test round.(contract_mps(A_mps), digits = 12) ==
              round.(contract_mps(F_mps), digits = 12)
        @test D_mps.sites != F_mps.sites
        # Ensure each right-normal site in B is actually right-normal
        #     Middle 3-leg tensors
        for i = 2:3
            site = F_mps.sites[i]
            @einsum res[i, j] := site[i, a, b] * site[j, a, b]
            @test round.(res, digits = 8) == [1 0; 0 1]
        end
        res = F_mps.sites[1] * transpose(F_mps.sites[1])
        @test round.(res, digits = 8) == [1 0; 0 1]

        # Bigger random test
        rank = 10
        A = rand(ComplexF64, (2 for _ = 1:rank)...)
        A_mps = mps(A)
        B_mps = set_orthogonality(A_mps, 2)
        right_edge = B_mps.sites[1]
        @test round.(right_edge * conj(transpose(right_edge)), digits = 8) == [1 0; 0 1]
        for i = 3:9
            site = B_mps.sites[i]
            conjugate = conj.(site)
            @einsum res[i, j] := site[a, b, i] * conjugate[a, b, j]
            @test round.(res, digits = 8) == [1 0; 0 1]
        end
        C_mps = set_orthogonality(B_mps, 7)
        for i in [1, 10]
            edge = B_mps.sites[i]
            @test round.(edge * conj(transpose(edge)), digits = 8) == [1 0; 0 1]
        end
        # Right normal sites
        for i = 2:6
            site = C_mps.sites[i]
            conjugate = conj.(site)
            @einsum res[i, j] := conjugate[i, a, b] * site[j, a, b]
            @test round.(res, digits = 8) == [1 0; 0 1]
        end
        # Left normal sites
        for i = 8:9
            site = C_mps.sites[i]
            conjugate = conj.(site)
            @einsum res[i, j] := site[a, b, i] * conjugate[a, b, j]
            @test round.(res, digits = 8) == [1 0; 0 1]
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
    @testset "Local operator expectation" begin
        # <σ_z>ᵩ for ϕ complex random
        rank = 5
        A = rand(ComplexF64, (2 for _ = 1:rank)...)
        A = A / sqrt(dot(A, A))
        A_mps = mps(A)
        A_mps_tensor = contract_mps(A_mps)
        σ_z = [1 0; 0 -1]
        I = [1 0; 0 1]
        conj_tensor = conj.(A_mps_tensor)
        @einsum full_res :=
            A_mps_tensor[a, b, α, c, d] * σ_z[α, β] * conj_tensor[a, b, β, c, d]
        easy_res = eval_local_op(A_mps, σ_z, 3)
        @test round(easy_res, digits = 12) == round(full_res, digits = 12)
    end
end

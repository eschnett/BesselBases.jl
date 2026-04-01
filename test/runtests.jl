using BesselBases
using Test

const B = BesselBases

using Printf

# @testset "Bessel functions" begin
#     for l in 0:5
#         for n in 1:5
#             @printf("  %4d  %4d  %12.6f\n", l, n, jl_zero(l, n))
#         end
#     end
# end

@testset "Bessel basis" begin
    # Effect of l on accuracy
    println("Orthogonality error  ‖G − I‖∞  (N=30, R=1, N_quad=2N+10)")
    println("-" ^ 50)
    @printf("  %4s  %8s  %12s  %12s\n", "l", "N_quad", "‖G−I‖∞", "deriv err")
    for l in [0, 1, 2, 5]
        N = 30
        R = 1.0
        b = BesselBasis(l, N, R)
        e_orth = B.check_orthogonality(b)
        e_deriv = B.check_derivative(b)
        @printf("  %4d  %8d  %12.2e  %12.2e\n", l, 2N+10, e_orth, e_deriv)
        @test abs(e_orth) <= 1.0e-11
        @test abs(e_deriv) <= 1.0e-12
    end

    println()
    println("Effect of N_quad (l=2, N=20, R=1)")
    println("-" ^ 50)
    @printf("  %8s  %12s\n", "N_quad", "‖G−I‖∞")
    for N_quad in [10, 20, 40, 50, 60]
        b = BesselBasis(2, 20, 1.0; N_quad)
        err = B.check_orthogonality(b)
        @printf("  %8d  %12.2e\n", N_quad, err)
        N_quad == 50 && @test abs(err) <= 1.0e-10
        N_quad == 60 && @test abs(err) <= 1.0e-13
    end
    # Expected: error drops below ε_mach around N_quad = 2N = 40

    println()
    println("Physical roundtrip (projection + reconstruction) error vs N")
    println("-" ^ 50)
    @printf("  %4s  %12s\n", "N", "‖Su−u‖/‖u‖")
    for N in [5, 10, 20, 40, 80, 160, 320]
        b = BesselBasis(2, N, 1.0)
        err = B.check_physical_roundtrip(b)
        @printf("  %4d  %12.2e\n", N, err)
        N == 320 && @test abs(err) <= 1.0e-11
    end
    # WRONG: Expected: spectral convergence (exponentially fast in N for smooth f)
    # Expected: superalgebraic convergence
end

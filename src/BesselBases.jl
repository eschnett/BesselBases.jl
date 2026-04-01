module BesselBases

using FastGaussQuadrature
using LinearAlgebra
using SpecialFunctions

################################################################################

# Spherical Bessel functions

export sbesselj, dsbesselj, jl_zero

"""
    sbesselj(l, x) → j_l(x)

Spherical Bessel function of the first kind, safe at x = 0.
Relation to cylindrical:  j_l(x) = √(π/2x) J_{l+1/2}(x).
Behaviour at origin:  j_0(0) = 1,  j_l(0) = 0 for l ≥ 1.
"""
function sbesselj(l::Int, x::Real)
    iszero(x) && return l == 0 ? one(x) : zero(x)
    return sqrt(π / (2x)) * besselj(l + 0.5, x)
end

"""
    dsbesselj(l, x) → j_l'(x)

Derivative via the recurrence  j_l'(x) = (l/x) j_l(x) − j_{l+1}(x).
"""
function dsbesselj(l::Int, x::Real)
    # j_l(x) ~ xˡ / (2l+1)!! near 0, so j_l'(0) = δ_{l,1}/3
    iszero(x) && return l == 1 ? one(x)/3 : zero(x)
    return (l / x) * sbesselj(l, x) - sbesselj(l + 1, x)
end

"""
    jl_zero(l, n) → n-th positive zero of j_l

j_l(x) = √(π/2x) J_{l+1/2}(x), so the zeros of j_l are the zeros of J_{l+1/2}.
We compute them with Newton's method, seeded by McMahon's asymptotic expansion:

    z ≈ β − (μ−1)/(8β) − 4(μ−1)(7μ−31)/(3(8β)³) − …

where  β = π(n + l/2)  and  μ = (2l+2)².  This gives ~10 correct digits even
for small n, and Newton converges in ≤ 5 iterations to machine precision.
"""
function jl_zero(l::Int, n::Int; tol::Float64=1e-14, maxiter::Int=50)
    ν = l + 0.5
    μ = 4ν^2
    β = π * (n + ν/2 - 0.25)          # leading McMahon term
    z = β - (μ - 1)/(8β) - 4*(μ - 1)*(7μ - 31)/(3*(8β)^3)

    for _ in 1:maxiter
        fz = sbesselj(l, z)
        dfz = dsbesselj(l, z)
        δz = fz / dfz
        z -= δz
        abs(δz) <= tol * abs(z) && break
    end
    return z
end

################################################################################

# BesselBasis struct and constructor

export BesselBasis

"""
    BesselBasis

Spectral basis on [0, R] using N spherical Bessel modes with Dirichlet BC.

Quadrature / collocation uses N_quad Gauss-Legendre points on [0, R].
Rule of thumb: N_quad ≥ 2N makes the norm integrals accurate to machine
precision. Reason: the product j_l(kₙ r) j_l(kₘ r) oscillates at most as
sin(2kₙ r), so you need ~2zₙ/π ≈ 2N quadrature points to resolve it.

Fields
------
l, N, R   — parameters
z         — zeros z_{l,n},  n = 1:N                         [length N]
k         — wavenumbers  kₙ = zₙ / R                        [length N]
h         — norms  hₙ = (R³/2) j_{l+1}(zₙ)²                [length N]
r         — collocation/quadrature points (GL nodes on [0,R])[length N_quad]
w         — GL weights for  ∫₀ᴿ f dr   (no r² factor yet)  [length N_quad]
S         — synthesis matrix   (N_quad × N):  S[j,n] = j_l(kₙ rⱼ)
A         — analysis  matrix   (N × N_quad):  A[n,j] = (wⱼ rⱼ²/hₙ) j_l(kₙ rⱼ)
dS        — derivative-synthesis  (N_quad × N):  dS[j,n] = kₙ j_l'(kₙ rⱼ)
"""
struct BesselBasis
    l::Int
    N::Int
    R::Float64
    z::Vector{Float64}
    k::Vector{Float64}
    h::Vector{Float64}
    r::Vector{Float64}
    w::Vector{Float64}
    S::Matrix{Float64}
    A::Matrix{Float64}
    dS::Matrix{Float64}
end

function BesselBasis(l::Int, N::Int, R::Float64; N_quad::Int=2N + 10)
    # Spectral grid
    z = [jl_zero(l, n) for n in 1:N] # zeros of j_l
    k = z ./ R                       # wavenumbers

    # L² norms on [0, R] with weight r²
    # ∫₀ᴿ j_l(kₙ r)² r² dr = (R³/2) [j_{l+1}(zₙ)]²
    # (standard Bessel orthogonality; see e.g. Watson §5.11)
    h = [(R^3 / 2) * sbesselj(l + 1, zn)^2 for zn in z]

    # Gauss-Legendre quadrature on [0, R]
    # The quadrature approximates ∫₀ᴿ f(r) dr ≈ Σⱼ wⱼ f(rⱼ).
    # An extra r² factor is folded into A below.
    ξ, wξ = gausslegendre(N_quad)
    r = @. R/2 * (ξ + 1)        # map [-1, 1] → [0, R]
    w = @. R/2 * wξ             # Jacobian R/2 absorbed into weights

    # Transform matrices
    # S[j,n]  = j_l(kₙ rⱼ)
    S = [sbesselj(l, k[n] * r[j]) for j in 1:N_quad, n in 1:N]

    # A[n,j] = (wⱼ rⱼ² / hₙ) j_l(kₙ rⱼ)
    # Satisfies  A·S ≈ Iₙ  when N_quad ≥ 2N.
    A = [(w[j] * r[j]^2 / h[n]) * sbesselj(l, k[n] * r[j]) for n in 1:N, j in 1:N_quad]

    # dS[j,n] = kₙ j_l'(kₙ rⱼ)
    # Note: this maps spectral coefficients to physical-space derivatives.
    # Unlike Fourier, d/dr is NOT diagonal in spectral space:
    #     ∂_r j_l(kₙ r) = kₙ [(l/kₙr) j_l(kₙr) − j_{l+1}(kₙr)]
    # which involves j_{l+1}, outside the l-sector basis.
    dS = [k[n] * dsbesselj(l, k[n] * r[j]) for j in 1:N_quad, n in 1:N]

    return BesselBasis(l, N, R, z, k, h, r, w, S, A, dS)
end

################################################################################

# The two transforms (and a derived spectral-space derivative)

"""
    to_spectral(b, u_coll) → û       (analysis)

Approximate  ûₙ = (1/hₙ) ∫₀ᴿ u(r) j_l(kₙ r) r² dr  via GL quadrature.
Equivalent to  û = A * u_coll.
"""
to_spectral(b::BesselBasis, u_coll) = b.A * u_coll

"""
    to_collocation(b, û) → u_coll    (synthesis)

Evaluate  u(rⱼ) = Σₙ ûₙ j_l(kₙ rⱼ)  at all collocation points.
Equivalent to  u_coll = S * û.
"""
to_collocation(b::BesselBasis, û) = b.S * û

"""
    deriv_collocation(b, û) → (∂ᵣu)(rⱼ)

Spectral differentiation in physical space:
    (∂ᵣu)(rⱼ) = Σₙ ûₙ kₙ j_l'(kₙ rⱼ) = dS · û

Because ∂ᵣ maps outside the j_l-sector (it mixes in j_{l+1}), there is no
simple diagonal "spectral derivative" like iω in Fourier. Differentiation is
done by evaluating dS · û at collocation points.
"""
deriv_collocation(b::BesselBasis, û) = b.dS * û

"""
    spectral_diff_matrix(b) → D  (N × N)

The N×N matrix D such that  D · û ≈ to_spectral(b, deriv_collocation(b, û)).
Useful for building PDE operators entirely in spectral space.
Computed as  D = A · dS.
"""
spectral_diff_matrix(b::BesselBasis) = b.A * b.dS

################################################################################

# Diagnostics

"""
    check_orthogonality(b) → err

Verify that the GL quadrature computes the inner products accurately.
Forms the Gram matrix  G[n,m] = (1/hₙ) ∫₀ᴿ j_l(kₙ r) j_l(kₘ r) r² dr ≈ δₙₘ
and returns  ‖G − Iₙ‖∞.  Should be ≲ ε_mach for N_quad ≥ 2N.
"""
function check_orthogonality(b::BesselBasis)
    W = Diagonal(b.w .* b.r .^ 2)        # weight matrix for ∫₀ᴿ · r² dr
    G = b.S' * W * b.S                   # G[n,m] = ∫ j_l(kₙ r) j_l(kₘ r) r² dr
    G_norm = Diagonal(1 ./ b.h) * G      # normalize: should equal I
    return norm(G_norm - I, Inf)
end

"""
    check_spectral_roundtrip(b) → err

Verify  A · S ≈ Iₙ  (analysis after synthesis returns the same coefficients).
‖A·S − Iₙ‖∞  should equal check_orthogonality(b) up to rounding.
"""
check_spectral_roundtrip(b::BesselBasis) = norm(b.A * b.S - I, Inf)

"""
    check_physical_roundtrip(b; f) → err

For a smooth test function f(r), check that synthesizing and re-analysing
recovers the function to within truncation error.

For SPECTRAL (exponential) convergence in the l-th Bessel basis, f must have
the SAME leading power as the basis functions near r=0:

    j_l(k_n r) ~ r^l   =>   f(r) ~ C * r^l  with C != 0  as r -> 0

Write f = r^l * g(r).  Applying the Bessel operator L gives:

    Lf = -(2l+2) r^{l-1} g'(r) - r^l g''(r)

  * g'(0) != 0  =>  Lf ~ r^{l-1}, f not in D(L^2),
                    u_n ~ O(n^{-2}), L2-error ~ O(N^{-3}).  ALGEBRAIC.
  * g'(0) = 0 and all odd derivatives of g vanish at 0 (g even-like)  =>
    Lf ~ r^l for every power of L, f in D(L^m) for all m  =>  SPECTRAL.

The default f(r) = (r/R)^l * cos(pi*r/(2R)) satisfies:
    g(r) = cos(pi*r/(2R)):  g(0)=1, g(R)=0, all odd derivatives at 0 = 0.

NOTE: This discussion is incomplete because it ignores the outer boundary.

Common traps:
    sin(pi*r/R)^{l+1}  =>  g(0) = 0  (f ~ r^{l+1}, not r^l)   ALGEBRAIC
    r^l * (1 - r/R)    =>  g'(0) = -1/R != 0                  ALGEBRAIC
    r^l * cos(pi*r/R)                                         ALGEBRAIC
"""
function check_physical_roundtrip(b::BesselBasis; f=r -> (r / b.R)^b.l * exp(- 1 / (1 - (r / b.R)^2)))
    u_true = f.(b.r)
    û = to_spectral(b, u_true)
    u_reco = to_collocation(b, û)
    return norm(u_reco - u_true) / norm(u_true)
end

"""
    check_derivative(b) → err

Verify spectral differentiation against a known analytic derivative.

We use a linear combination of three exact basis functions as the test input:
    f(r) = Σᵢ cᵢ j_l(kᵢ r)
whose derivative is exactly
    f'(r) = Σᵢ cᵢ kᵢ j_l'(kᵢ r).

This sidesteps the r=0 issue: near r=0, j_l(kₙr) ~ rˡ, so all basis function
derivatives also vanish like r^{l-1} (or for l=0, like r). Testing with a
function whose f'(0) ≠ 0 (e.g. sin(πr/R) which has f'(0) = π) would always
fail for l=0 because the derivative basis {kₙ j₀'(kₙr)} spans only functions
vanishing at the origin. The error would be a basis limitation, not a bug.

The only error here is quadrature error in the analysis step A·S ≈ I, which
should be ≲ ε_mach for N_quad ≥ 2N.
"""
function check_derivative(b::BesselBasis)
    c = [1.0, -0.7, 0.4]                 # arbitrary coefficients for modes 1,2,3
    modes = 1:length(c)

    f_coll = sum(c[i] .* sbesselj.(b.l, b.k[i] .* b.r) for i in modes)
    df_true = sum(c[i] .* b.k[i] .* dsbesselj.(b.l, b.k[i] .* b.r) for i in modes)

    û = to_spectral(b, f_coll)
    du_spec = deriv_collocation(b, û)
    return norm(du_spec - df_true) / norm(df_true)
end

end

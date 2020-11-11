import TridiagonalMatrixAlgorithm


struct LinearPropagatorTDMA{
    T<:AbstractFloat,
    C<:Complex{T},
    A<:AbstractArray{T},
    AC<:AbstractArray{Complex{T}},
} <: LinearPropagator
    Rr :: C
    r :: A
    a :: AC
    b :: AC
    c :: AC
    d :: AC
    tmp :: AC
end


function LinearPropagator(
    unit::Units.UnitR,
    grid::Grids.GridRn,
    medium::Media.Medium,
    field::Fields.Field,
    guard::Guards.Guard,
    PARAXIAL::Bool,
)
    @assert PARAXIAL

    Nr = grid.Nr
    r = grid.r

    w0 = field.w0
    beta0 = Media.beta_func(medium, w0)

    Rr = 1im / (2 * beta0) * unit.z / unit.r^2
    Rr = convert(Complex{FloatGPU}, Rr)

    # Tridiagonal matrix coefficients:
    a = zeros(Complex{FloatGPU}, Nr)
    b = zeros(Complex{FloatGPU}, Nr)
    c = zeros(Complex{FloatGPU}, Nr)

    d = zeros(Complex{FloatGPU}, Nr)   # right-hand-side vector
    tmp = zeros(Complex{FloatGPU}, Nr)   # array for temporary storage

    return LinearPropagatorTDMA(Rr, r, a, b, c, d, tmp)
end


function propagate!(
    Egpu::AbstractArray{Complex{T}},
    LP::LinearPropagatorTDMA,
    dz::T
) where T
    E = Array(Egpu)

    Rr = LP.Rr
    r = LP.r
    a = LP.a
    b = LP.b
    c = LP.c
    d = LP.d
    tmp = LP.tmp

    N = length(E)

    h0  = (r[3] - r[2]) / 2
    ksi = 3/2 * (dz * Rr) *
          (r[2] + h0) / (r[2] - h0) / (r[2]^2 + h0 * (r[2] + h0) - h0^2)

    # Tridiagonal matrix coefficients:
    b[1] = 1
    c[1] = -ksi / (1 + ksi)
    for i=2:N-1
        a[i] = (r[i] + r[i-1]) / (r[i] - r[i-1])
        c[i] = (r[i+1] + r[i]) / (r[i+1] - r[i])
        b[i] = -a[i] - c[i] - 2/3 / (dz * Rr) *
               (r[i+1]^2 + r[i] * (r[i+1] - r[i-1]) - r[i-1]^2)
    end
    a[N] = 0
    b[N] = 1

    # Right-hand-side vector:
    d[1] = (ksi * E[2] + (1 - ksi) * E[1]) / (1 + ksi)
    for i=2:N-1
        d[i] = -a[i] * E[i-1] - conj(b[i]) * E[i] - c[i] * E[i+1]
    end
    d[N] = 0

    # Solve:
    TridiagonalMatrixAlgorithm.tridag!(E, a, b, c, d, tmp)
    # @. E = E * exp(1im * k0 * (dz * zu))   # constant phase

    E = CUDA.CuArray(E)
    @. Egpu = E

    return nothing
end

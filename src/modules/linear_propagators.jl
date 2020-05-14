const TSPlan = Union{HankelTransforms.Plan, FourierTransforms.Plan, Nothing}


struct LinearPropagator{
    T<:AbstractFloat,
    A<:AbstractArray{Complex{T}},
    G<:Guards.Guard,
    P<:TSPlan
}
    KZ :: A
    guard :: G
    TS :: P
end


function propagate!(
    E::AbstractArray{Complex{T}},
    LP::LinearPropagator,
    z::T
) where T
    forward_transform_space!(E, LP.TS)
    @. E = E * exp(-1im * LP.KZ * z)
    Guards.apply_spectral_filter!(E, LP.guard)
    inverse_transform_space!(E, LP.TS)
    return nothing
end


# ******************************************************************************
function LinearPropagator(
    unit::Units.UnitR,
    grid::Grids.GridR,
    medium::Media.Medium,
    field::Fields.FieldR,
    guard::Guards.Guard,
    PARAXIAL::Bool,
)
    w0 = field.w0
    vf = Media.group_velocity(medium, w0)   # frame velocity

    KZ = zeros(ComplexF64, grid.Nr)
    for i=1:grid.Nr
        kt = grid.k[i] * unit.k
        KZ[i] = Kfunc(PARAXIAL, medium, w0, kt) * unit.z
        # Here the moving frame is added to reduce the truncation error, which
        # appears due to use of Float32 precision:
        KZ[i] = KZ[i] - w0 / vf * unit.z
        KZ[i] = conj(KZ[i])   # in order to make fft instead of ifft
    end
    KZ = CuArrays.CuArray{Complex{FloatGPU}}(KZ)

    return LinearPropagator(KZ, guard, field.HT)
end


function LinearPropagator(
    unit::Units.UnitT,
    grid::Grids.GridT,
    medium::Media.Medium,
    field::Fields.FieldT,
    guard::Guards.Guard,
    PARAXIAL::Bool,
)
    w0 = field.w0
    vf = Media.group_velocity(medium, w0)   # frame velocity

    KZ = zeros(ComplexF64, grid.Nt)
    for i=1:grid.Nt
        w = grid.w[i] * unit.w
        KZ[i] = Kfunc(PARAXIAL, medium, w, 0.0) * unit.z
        KZ[i] = KZ[i] - w / vf * unit.z
        KZ[i] = conj(KZ[i])   # in order to make fft instead of ifft
    end

    return LinearPropagator(KZ, guard, nothing)
end


function LinearPropagator(
    unit::Units.UnitRT,
    grid::Grids.GridRT,
    medium::Media.Medium,
    field::Fields.FieldRT,
    guard::Guards.Guard,
    PARAXIAL::Bool,
)
    w0 = field.w0
    vf = Media.group_velocity(medium, w0)   # frame velocity

    KZ = zeros(ComplexF64, (grid.Nr, grid.Nt))
    for j=1:grid.Nt
    for i=1:grid.Nr
        kt = grid.k[i] * unit.k
        w = grid.w[j] * unit.w
        KZ[i, j] = Kfunc(PARAXIAL, medium, w, kt) * unit.z
        KZ[i, j] = KZ[i, j] - w / vf * unit.z
        KZ[i, j] = conj(KZ[i, j])   # in order to make fft instead of ifft
    end
    end
    KZ = CuArrays.CuArray{Complex{FloatGPU}}(KZ)

    return LinearPropagator(KZ, guard, field.HT)
end


function LinearPropagator(
    unit::Units.UnitXY,
    grid::Grids.GridXY,
    medium::Media.Medium,
    field::Fields.FieldXY,
    guard::Guards.Guard,
    PARAXIAL::Bool,
)
    w0 = field.w0
    vf = Media.group_velocity(medium, w0)   # frame velocity

    KZ = zeros(ComplexF64, (grid.Nx, grid.Ny))
    for j=1:grid.Ny
    for i=1:grid.Nx
        kt = sqrt((grid.kx[i] * unit.kx)^2 + (grid.ky[j] * unit.ky)^2)
        KZ[i, j] = Kfunc(PARAXIAL, medium, w0, kt) * unit.z
        # Here the moving frame is added to reduce the truncation error, which
        # appears due to use of Float32 precision:
        KZ[i, j] = KZ[i, j] - w0 / vf * unit.z
        KZ[i, j] = conj(KZ[i, j])   # in order to make fft instead of ifft
    end
    end
    KZ = CuArrays.CuArray{Complex{FloatGPU}}(KZ)

    return LinearPropagator(KZ, guard, field.FT)
end


# ******************************************************************************
function Kfunc(PARAXIAL, medium, w, kt)
    if PARAXIAL
        K = Kfunc_paraxial(medium, w, kt)
    else
        K = Kfunc_nonparaxial(medium, w, kt)
    end
    return K
end


function Kfunc_paraxial(medium, w, kt)
    beta = Media.beta_func(medium, w)
    if beta != 0
        K = beta - kt^2 / (2 * beta)
    else
        K = zero(kt)
    end
    return K
end


function Kfunc_nonparaxial(medium, w, kt)
    beta = Media.beta_func(medium, w)
    return sqrt(beta^2 - kt^2 + 0im)
end

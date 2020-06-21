struct LinearPropagator{
    T<:AbstractFloat,
    A<:AbstractArray{Complex{T}},
    G<:Guards.Guard,
    P<:Union{HankelTransforms.Plan, FFTW.Plan, Nothing}
}
    KZ :: A
    guard :: G
    PS :: P
end


function propagate!(
    E::AbstractArray{Complex{T}},
    LP::LinearPropagator,
    z::T
) where T
    forward_transform_space!(E, LP.PS)
    @. E = E * exp(1im * LP.KZ * z)
    Guards.apply_spectral_filter!(E, LP.guard)
    inverse_transform_space!(E, LP.PS)
    return nothing
end


# ******************************************************************************
function LinearPropagator(
    unit::Units.UnitR,
    grid::Grids.GridR,
    medium::Media.Medium,
    field::Fields.Field,
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
    end
    KZ = CUDA.CuArray{Complex{FloatGPU}}(KZ)

    return LinearPropagator(KZ, guard, field.PS)
end


function LinearPropagator(
    unit::Units.UnitT,
    grid::Grids.GridT,
    medium::Media.Medium,
    field::Fields.Field,
    guard::Guards.Guard,
    PARAXIAL::Bool,
)
    w0 = field.w0
    vf = Media.group_velocity(medium, w0)   # frame velocity

    KZ = zeros(ComplexF64, grid.Nt)
    for i=1:grid.Nt
        kt = 0.0
        w = grid.w[i] * unit.w
        KZ[i] = Kfunc(PARAXIAL, medium, w, kt) * unit.z
        KZ[i] = KZ[i] - w / vf * unit.z
    end

    return LinearPropagator(KZ, guard, field.PS)
end


function LinearPropagator(
    unit::Units.UnitRT,
    grid::Grids.GridRT,
    medium::Media.Medium,
    field::Fields.Field,
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
    end
    end
    KZ = CUDA.CuArray{Complex{FloatGPU}}(KZ)

    return LinearPropagator(KZ, guard, field.PS)
end


function LinearPropagator(
    unit::Units.UnitXY,
    grid::Grids.GridXY,
    medium::Media.Medium,
    field::Fields.Field,
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
    end
    end
    KZ = CUDA.CuArray{Complex{FloatGPU}}(KZ)

    return LinearPropagator(KZ, guard, field.PS)
end


function LinearPropagator(
    unit::Units.UnitXYT,
    grid::Grids.GridXYT,
    medium::Media.Medium,
    field::Fields.Field,
    guard::Guards.Guard,
    PARAXIAL::Bool,
)
    w0 = field.w0
    vf = Media.group_velocity(medium, w0)   # frame velocity

    KZ = zeros(ComplexF64, (grid.Nx, grid.Ny, grid.Nt))
    for k=1:grid.Nt
    for j=1:grid.Ny
    for i=1:grid.Nx
        kt = sqrt((grid.kx[i] * unit.kx)^2 + (grid.ky[j] * unit.ky)^2)
        w = grid.w[k] * unit.w
        KZ[i, j, k] = Kfunc(PARAXIAL, medium, w, kt) * unit.z
        KZ[i, j, k] = KZ[i, j, k] - w / vf * unit.z
    end
    end
    end
    KZ = CUDA.CuArray{Complex{FloatGPU}}(KZ)

    return LinearPropagator(KZ, guard, field.PS)
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

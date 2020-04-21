module LinearPropagators

import CuArrays
import HankelTransforms

import Constants: FloatGPU
import Fields
import FourierTransforms
import Grids
import Guards
import Media
import Units


abstract type LinearPropagator end


struct LinearPropagatorNone{
    T<:AbstractFloat,
    A<:AbstractArray{Complex{T}},
    G<:Guards.Guard,
} <: LinearPropagator
    KZ :: A
    guard :: G
end


struct LinearPropagatorHankel{
    T<:AbstractFloat,
    A<:AbstractArray{Complex{T}},
    G<:Guards.Guard,
    P<:HankelTransforms.Plan
} <: LinearPropagator
    KZ :: A
    guard :: G
    HT :: P
end


struct LinearPropagatorFourier{
    T<:AbstractFloat,
    A<:AbstractArray{Complex{T}},
    G<:Guards.Guard,
    P<:FourierTransforms.Plan
} <: LinearPropagator
    KZ :: A
    guard :: G
    FT :: P
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

    return LinearPropagatorHankel(KZ, guard, field.HT)
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

    return LinearPropagatorNone(KZ, guard)
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

    return LinearPropagatorHankel(KZ, guard, field.HT)
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

    return LinearPropagatorFourier(KZ, guard, field.FT)
end


# ******************************************************************************
function propagate!(
    E::AbstractArray{Complex{T}, 1},
    LP::LinearPropagatorNone,
    z::T
) where T
    @. E = E * exp(-1im * LP.KZ * z)
    Guards.apply_spectral_filter!(E, LP.guard)
    return nothing
end


function propagate!(
    E::AbstractArray{Complex{T}},
    LP::LinearPropagatorHankel,
    z::T
) where T
    HankelTransforms.dht!(E, LP.HT)
    @. E = E * exp(-1im * LP.KZ * z)
    Guards.apply_spectral_filter!(E, LP.guard)
    HankelTransforms.idht!(E, LP.HT)
    return nothing
end


function propagate!(
    E::AbstractArray{Complex{T}},
    LP::LinearPropagatorFourier,
    z::T
) where T
    FourierTransforms.fft!(E, LP.FT)
    @. E = E * exp(-1im * LP.KZ * z)
    Guards.apply_spectral_filter!(E, LP.guard)
    FourierTransforms.ifft!(E, LP.FT)
    return nothing
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


end

module LinearPropagators

import CuArrays
import HankelTransforms

import Constants: FloatGPU
import Fields
import Fourier
import Grids
import Guards
import Media
import Units


abstract type LinearPropagator end


struct LinearPropagatorR{T} <: LinearPropagator
    KZ :: AbstractArray{Complex{T}, 1}
    HT :: HankelTransforms.Plan
    guard :: Guards.Guard
end


struct LinearPropagatorT{T} <: LinearPropagator
    KZ :: AbstractArray{Complex{T}, 1}
    FT :: Fourier.FourierTransform
    guard :: Guards.Guard
end


struct LinearPropagatorRT{T} <: LinearPropagator
    KZ :: AbstractArray{Complex{T}, 2}
    HT :: HankelTransforms.Plan
    guard :: Guards.Guard
end


struct LinearPropagatorXY{T} <: LinearPropagator
    KZ :: AbstractArray{Complex{T}, 2}
    FT :: Fourier.FourierTransform
    guard :: Guards.Guard
end


function LinearPropagator(
    unit::Units.UnitR,
    grid::Grids.GridR,
    medium::Media.Medium,
    field::Fields.Field,
    guard::Guards.Guard,
    keys::NamedTuple,
)
    KPARAXIAL = keys.KPARAXIAL

    beta = Media.beta_func(medium, field.w0)

    if KPARAXIAL
        KZ = @. beta - (grid.k * unit.k)^2 / (2 * beta)
    else
        KZ = @. sqrt(beta^2 - (grid.k * unit.k)^2 + 0im)
    end

    # In order to reduce the truncation error, which appears due to use of
    # Float32 precision, perform the calculations in a moving frame:
    vf = Media.group_velocity(medium, field.w0)   # frame velocity
    @. KZ = (KZ - field.w0 / vf) * unit.z

    @. KZ = conj(KZ)
    KZ = CuArrays.CuArray{Complex{FloatGPU}}(KZ)

    return LinearPropagatorR(KZ, grid.HT, guard)
end


function LinearPropagator(
    unit::Units.UnitT,
    grid::Grids.GridT,
    medium::Media.Medium,
    field::Fields.Field,
    guard::Guards.Guard,
    keys::NamedTuple,
)
    beta = Media.beta_func.(Ref(medium), grid.w * unit.w)
    vf = Media.group_velocity(medium, field.w0)   # frame velocity
    KZ = @. beta + 0im
    @. KZ = (KZ - grid.w * unit.w / vf) * unit.z
    @. KZ = conj(KZ)
    return LinearPropagatorT(KZ, grid.FT, guard)
end


function LinearPropagator(
    unit::Units.UnitRT,
    grid::Grids.GridRT,
    medium::Media.Medium,
    field::Fields.Field,
    guard::Guards.Guard,
    keys::NamedTuple,
)
    KPARAXIAL = keys.KPARAXIAL

    beta = Media.beta_func.(Ref(medium), grid.w * unit.w)

    KZ = zeros(ComplexF64, (grid.Nr, grid.Nw))
    if KPARAXIAL
        for iw=1:grid.Nw
            if beta[iw] != 0
                for ir=1:grid.Nr
                    KZ[ir, iw] = beta[iw] -
                                 (grid.k[ir] * unit.k)^2 / (2 * beta[iw])
                end
            end
        end
    else
        for iw=1:grid.Nw
        for ir=1:grid.Nr
            KZ[ir, iw] = sqrt(beta[iw]^2 - (grid.k[ir] * unit.k)^2 + 0im)
        end
        end
    end

    vf = Media.group_velocity(medium, field.w0)   # frame velocity
    for iw=1:grid.Nw
    for ir=1:grid.Nr
        KZ[ir, iw] = (KZ[ir, iw] - grid.w[iw] * unit.w / vf) * unit.z
    end
    end
    @. KZ = conj(KZ)
    KZ = CuArrays.CuArray{Complex{FloatGPU}}(KZ)

    return LinearPropagatorRT(KZ, grid.HT, guard)
end


function LinearPropagator(
    unit::Units.UnitXY,
    grid::Grids.GridXY,
    medium::Media.Medium,
    field::Fields.Field,
    guard::Guards.Guard,
    keys::NamedTuple,
)
    KPARAXIAL = keys.KPARAXIAL

    beta = Media.beta_func(medium, field.w0)

    KZ = zeros(ComplexF64, (grid.Nx, grid.Ny))
    if KPARAXIAL
        for iy=1:grid.Ny
        for ix=1:grid.Nx
            KZ[ix, iy] = beta - ((grid.kx[ix] * unit.kx)^2 +
                                 (grid.ky[iy] * unit.ky)^2) / (2 * beta)
        end
        end
    else
        for iy=1:grid.Ny
        for ix=1:grid.Nx
            KZ[ix, iy] = sqrt(beta^2 - ((grid.kx[ix] * unit.kx)^2 +
                                        (grid.ky[iy] * unit.ky)^2) + 0im)
        end
        end
    end

    # In order to reduce the truncation error, which appears due to use of
    # Float32 precision, perform the calculations in a moving frame:
    vf = Media.group_velocity(medium, field.w0)   # frame velocity
    @. KZ = (KZ - field.w0 / vf) * unit.z

    @. KZ = conj(KZ)
    KZ = CuArrays.CuArray{Complex{FloatGPU}}(KZ)

    return LinearPropagatorXY(KZ, grid.FT, guard)
end


function propagate!(
    E::AbstractArray{Complex{T}, 1},
    LP::LinearPropagatorR,
    z::T
) where T
    HankelTransforms.dht!(E, LP.HT)
    @. E = E * exp(-1im * LP.KZ * z)
    Guards.apply_spectral_filter!(E, LP.guard)
    HankelTransforms.idht!(E, LP.HT)
    return nothing
end


function propagate!(
    S::AbstractArray{Complex{T}, 1},
    LP::LinearPropagatorT,
    z::T
) where T
    @. S = S * exp(-1im * LP.KZ * z)
    Guards.apply_spectral_filter!(S, LP.guard)
    return nothing
end


function propagate!(
    E::AbstractArray{Complex{T}, 2},
    LP::LinearPropagatorRT,
    z::T
) where T
    HankelTransforms.dht!(E, LP.HT)
    @. E = E * exp(-1im * LP.KZ * z)
    Guards.apply_spectral_filter!(E, LP.guard)
    HankelTransforms.idht!(E, LP.HT)
    return nothing
end


function propagate!(
    E::AbstractArray{Complex{T}, 2},
    LP::LinearPropagatorXY,
    z::T
) where T
    Fourier.fft!(E, LP.FT)
    @. E = E * exp(-1im * LP.KZ * z)
    Guards.apply_spectral_filter!(E, LP.guard)
    Fourier.ifft!(E, LP.FT)
    return nothing
end


end

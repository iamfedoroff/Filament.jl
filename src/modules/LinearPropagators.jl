module LinearPropagators

import CuArrays
import CUDAnative
import FFTW
import LinearAlgebra

import Grids
import Media
import Units

const FloatGPU = Float32


abstract type LinearPropagator end


struct LinearPropagatorR{T} <: LinearPropagator
    KZ :: AbstractArray{Complex{T}, 1}
end


struct LinearPropagatorRT <: LinearPropagator
end


struct LinearPropagatorXY{T} <: LinearPropagator
    KZ :: AbstractArray{Complex{T}, 2}
    pfft :: FFTW.Plan
    pifft :: FFTW.Plan
end


function LinearPropagator(unit::Units.UnitR,
                          grid::Grids.GridR,
                          medium::Media.Medium,
                          w0::AbstractFloat,
                          PARAXIAL::Bool)
    beta = Media.beta_func(medium, w0)
    if PARAXIAL
        KZ = @. beta - (grid.k * unit.k)^2 / (2 * beta)
    else
        KZ = @. sqrt(beta^2 - (grid.k * unit.k)^2 + 0im)
    end
    @. KZ = KZ * unit.z
    @. KZ = conj(KZ)
    KZ = CuArrays.CuArray(convert(Array{Complex{FloatGPU}, 1}, KZ))

    return LinearPropagatorR(KZ)
end


function LinearPropagator(unit::Units.UnitXY,
                          grid::Grids.GridXY,
                          medium::Media.Medium,
                          w0::AbstractFloat,
                          PARAXIAL::Bool)
    beta = Media.beta_func(medium, w0)

    KZ = zeros(ComplexF64, (grid.Nx, grid.Ny))
    if PARAXIAL
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
    @. KZ = KZ * unit.z
    @. KZ = conj(KZ)
    KZ = CuArrays.CuArray(convert(Array{Complex{FloatGPU}, 2}, KZ))

    pfft = FFTW.plan_fft(KZ)
    pifft = FFTW.plan_ifft(KZ)

    return LinearPropagatorXY(KZ, pfft, pifft)
end


function propagate!(E::AbstractArray{Complex{T}, 1},
                    LP::LinearPropagatorR,
                    z::T) where T
    # Hankel.dht!(grid.HT, field.E)
    # @. field.E = field.E * CUDAnative.exp(-1im * model.KZ * dz)
    # Guards.apply_spectral_filter!(guard, field.E)
    # Hankel.idht!(grid.HT, field.E)
    return nothing
end


function propagate!(E::AbstractArray{Complex{T}, 2},
                    LP::LinearPropagatorXY,
                    z::T) where T<:AbstractFloat
    LinearAlgebra.mul!(E, LP.pfft, E)   # fft!(E)
    @. E = E * myexp(-1im * LP.KZ * z)
    # Guards.apply_spectral_filter!(E, guard)
    LinearAlgebra.mul!(E, LP.pifft, E)   # ifft(E)
    return nothing
end


"""
Complex version of CUDAnative.exp function. Adapted from
https://discourse.julialang.org/t/base-function-in-cuda-kernels/21866/4
"""
# @inline function CUDAnative.exp(x::Complex{T}) where T
@inline function myexp(x::Complex{T}) where T
    scale = CUDAnative.exp(x.re)
    return Complex{T}(scale * CUDAnative.cos(x.im),
                      scale * CUDAnative.sin(x.im))
end


end

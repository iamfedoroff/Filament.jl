module Fields

import CuArrays
import HankelTransforms

import AnalyticSignals
import Constants: FloatGPU, C0, HBAR
import FourierTransforms
import Grids
import Units


abstract type Field end


# ******************************************************************************
# R
# ******************************************************************************
struct FieldR{
    T<:AbstractFloat,
    AC<:AbstractArray{Complex{T}},
    PH<:HankelTransforms.Plan,
} <: Field
    w0 :: T
    E :: AC
    HT :: PH
end


function Field(
    unit::Units.UnitR,
    grid::Grids.GridR,
    lam0::T,
    initial_condition::Function,
) where T<:AbstractFloat
    w0 = convert(T, 2 * pi * C0 / lam0)

    E = initial_condition(grid.r, unit.r, unit.I)
    E = CuArrays.CuArray{Complex{T}}(E)

    HT = HankelTransforms.plan(grid.rmax, E)
    return FieldR(w0, E, HT)
end


# ******************************************************************************
# T
# ******************************************************************************
struct FieldT{
    T<:AbstractFloat,
    A<:AbstractArray{T},
    AC<:AbstractArray{Complex{T}},
    PF <: FourierTransforms.FourierTransform,
} <: Field
    w0 :: T
    E :: AC
    rho :: A
    kdrho :: A
    FT :: PF
end


function Field(
    unit::Units.UnitT,
    grid::Grids.GridT,
    lam0::T,
    initial_condition::Function,
) where T<:AbstractFloat
    w0 = convert(T, 2 * pi * C0 / lam0)

    FT = FourierTransforms.FourierTransformT(grid.Nt)

    E = initial_condition(grid.t, unit.t, unit.I)
    E = Array{Complex{T}}(E)
    AnalyticSignals.rsig2asig!(E, FT)   # convert to analytic signal

    rho = zeros(T, grid.Nt)
    kdrho = zeros(T, grid.Nt)

    # Initialize a dummy GPU array in order to trigger the creation of the
    # device context. This will allow to call CUDAdrv.synchronize() in the
    # main cycle.
    tmp = CuArrays.zeros(T, 1)

    return FieldT(w0, E, rho, kdrho, FT)
end


# ******************************************************************************
# RT
# ******************************************************************************
struct FieldRT{
    T<:AbstractFloat,
    A<:AbstractArray{T},
    AC<:AbstractArray{Complex{T}},
    PH<:HankelTransforms.Plan,
    PF<:FourierTransforms.FourierTransform,
} <: Field
    w0 :: T
    E :: AC
    rho :: A
    kdrho :: A
    HT :: PH
    FT :: PF
end


function Field(
    unit::Units.UnitRT,
    grid::Grids.GridRT,
    lam0::T,
    initial_condition::Function,
) where T<:AbstractFloat
    w0 = convert(T, 2 * pi * C0 / lam0)

    FT = FourierTransforms.FourierTransformRT(grid.Nr, grid.Nt)

    E = initial_condition(grid.r, grid.t, unit.r, unit.t, unit.I)
    E = CuArrays.CuArray{Complex{T}}(E)
    AnalyticSignals.rsig2asig!(E, FT)   # convert to analytic signal

    rho = CuArrays.zeros(T, (grid.Nr, grid.Nt))
    kdrho = CuArrays.zeros(T, (grid.Nr, grid.Nt))

    Nthalf = AnalyticSignals.half(grid.Nt)
    region = CartesianIndices((grid.Nr, Nthalf))
    HT = HankelTransforms.plan(grid.rmax, E, region)
    return FieldRT(w0, E, rho, kdrho, HT, FT)
end


# ******************************************************************************
# XY
# ******************************************************************************
struct FieldXY{
    T<:AbstractFloat,
    AC<:AbstractArray{Complex{T}},
    PF<:FourierTransforms.FourierTransform
} <: Field
    w0 :: T
    E :: AC
    FT :: PF
end


function Field(
    unit::Units.UnitXY,
    grid::Grids.GridXY,
    lam0::T,
    initial_condition::Function,
) where T<:AbstractFloat
    w0 = convert(T, 2 * pi * C0 / lam0)

    E = initial_condition(grid.x, grid.y, unit.x, unit.y, unit.I)
    E = CuArrays.CuArray{Complex{T}}(E)

    FT = FourierTransforms.FourierTransformXY(grid.Nx, grid.Ny)
    return FieldXY(w0, E, FT)
end


end

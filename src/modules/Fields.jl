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


function Field(unit::Units.UnitR, grid::Grids.GridR, p::Tuple)
    lam0, initial_condition, HTLOAD, file_ht = p
    T = typeof(lam0)

    w0 = convert(T, 2 * pi * C0 / lam0)

    E = initial_condition(grid.r, unit.r, unit.I)
    E = CuArrays.CuArray{Complex{T}}(E)

    if HTLOAD
        HT = HankelTransforms.plan(file_ht)
    else
        HT = HankelTransforms.plan(grid.rmax, E, save=true, fname="ht.jld2")
    end
    return FieldR(w0, E, HT)
end


# ******************************************************************************
# T
# ******************************************************************************
struct FieldT{
    T<:AbstractFloat,
    A<:AbstractArray{T},
    AC<:AbstractArray{Complex{T}},
    PF <: FourierTransforms.Plan,
} <: Field
    w0 :: T
    E :: AC
    rho :: A
    kdrho :: A
    FT :: PF
end


function Field(unit::Units.UnitT, grid::Grids.GridT, p::Tuple)
    lam0, initial_condition = p
    T = typeof(lam0)

    w0 = convert(T, 2 * pi * C0 / lam0)

    E = initial_condition(grid.t, unit.t, unit.I)
    E = Array{Complex{T}}(E)

    FT = FourierTransforms.Plan(E)
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
    PF<:FourierTransforms.Plan,
} <: Field
    w0 :: T
    E :: AC
    rho :: A
    kdrho :: A
    HT :: PH
    FT :: PF
end


function Field(unit::Units.UnitRT, grid::Grids.GridRT, p::Tuple)
    lam0, initial_condition, HTLOAD, file_ht = p
    T = typeof(lam0)

    w0 = convert(T, 2 * pi * C0 / lam0)

    E = initial_condition(grid.r, grid.t, unit.r, unit.t, unit.I)
    E = CuArrays.CuArray{Complex{T}}(E)

    FT = FourierTransforms.Plan(E, [2])
    AnalyticSignals.rsig2asig!(E, FT)   # convert to analytic signal

    if HTLOAD
        HT = HankelTransforms.plan(file_ht)
    else
        Nthalf = AnalyticSignals.half(grid.Nt)
        region = CartesianIndices((grid.Nr, Nthalf))
        HT = HankelTransforms.plan(
            grid.rmax, E, region, save=true, fname="ht.jld2",
        )
    end

    rho = CuArrays.zeros(T, (grid.Nr, grid.Nt))
    kdrho = CuArrays.zeros(T, (grid.Nr, grid.Nt))
    return FieldRT(w0, E, rho, kdrho, HT, FT)
end


# ******************************************************************************
# XY
# ******************************************************************************
struct FieldXY{
    T<:AbstractFloat,
    AC<:AbstractArray{Complex{T}},
    PF<:FourierTransforms.Plan
} <: Field
    w0 :: T
    E :: AC
    FT :: PF
end


function Field(unit::Units.UnitXY, grid::Grids.GridXY, p::Tuple)
    lam0, initial_condition = p
    T = typeof(lam0)

    w0 = convert(T, 2 * pi * C0 / lam0)

    E = initial_condition(grid.x, grid.y, unit.x, unit.y, unit.I)
    E = CuArrays.CuArray{Complex{T}}(E)

    FT = FourierTransforms.Plan(E)
    return FieldXY(w0, E, FT)
end


end

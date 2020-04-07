module Fields

import CuArrays
import HankelTransforms

import Constants: FloatGPU, C0, HBAR
import Fourier
import Grids
import Units


abstract type Field end


struct FieldR{
    T<:AbstractFloat,
    UC<:AbstractArray{Complex{T}},
    PH<:HankelTransforms.Plan,
} <: Field
    w0 :: T
    E :: UC
    HT :: PH
end


struct FieldT{
    T<:AbstractFloat,
    UC<:AbstractArray{Complex{T}},
    UF<:AbstractArray{T},
    PF <: Fourier.FourierTransform,
} <: Field
    w0 :: T
    E :: UC
    S :: UC
    rho :: UF
    kdrho :: UF
    FT :: PF
end


struct FieldRT{
    T<:AbstractFloat,
    UC<:AbstractArray{Complex{T}},
    UF<:AbstractArray{T},
    PH<:HankelTransforms.Plan,
    PF<:Fourier.FourierTransform,
} <: Field
    w0 :: T
    E :: UC
    S :: UC
    rho :: UF
    kdrho :: UF
    HT :: PH
    FT :: PF
end


struct FieldXY{
    T<:AbstractFloat,
    UC<:AbstractArray{Complex{T}},
    PF<:Fourier.FourierTransform
} <: Field
    w0 :: T
    E :: UC
    FT :: PF
end


function Field(
    unit::Units.UnitR,
    grid::Grids.GridR,
    lam0::T,
    initial_condition::Function,
) where T<:AbstractFloat
    w0 = 2 * pi * C0 / lam0
    w0 = convert(T, w0)

    E = initial_condition(grid.r, unit.r, unit.I)
    E = CuArrays.CuArray{Complex{T}}(E)

    HT = HankelTransforms.plan(grid.rmax, E)
    return FieldR(w0, E, HT)
end


function Field(
    unit::Units.UnitT,
    grid::Grids.GridT,
    lam0::T,
    initial_condition::Function,
) where T<:AbstractFloat
    w0 = 2 * pi * C0 / lam0
    w0 = convert(T, w0)

    E = initial_condition(grid.t, unit.t, unit.I)
    E = Array{Complex{T}}(E)

    FT = Fourier.FourierTransformT(grid.Nt)

    S = zeros(Complex{T}, grid.Nw)
    Fourier.rfft!(S, FT, E)   # time -> frequency

    Fourier.hilbert!(E, FT, S)   # spectrum real to signal analytic

    rho = zeros(T, grid.Nt)
    kdrho = zeros(T, grid.Nt)

    # Initialize a dummy GPU array in order to trigger the creation of the
    # device context. This will allow to call CUDAdrv.synchronize() in the
    # main cycle.
    tmp = CuArrays.zeros(T, 1)

    return FieldT(w0, E, S, rho, kdrho, FT)
end


function Field(
    unit::Units.UnitRT,
    grid::Grids.GridRT,
    lam0::T,
    initial_condition::Function,
) where T<:AbstractFloat
    w0 = 2 * pi * C0 / lam0
    w0 = convert(T, w0)

    E = initial_condition(grid.r, grid.t, unit.r, unit.t, unit.I)
    E = CuArrays.CuArray{Complex{T}}(E)

    FT = Fourier.FourierTransformRT(grid.Nr, grid.Nt)

    S = CuArrays.zeros(Complex{T}, (grid.Nr, grid.Nw))
    Fourier.rfft!(S, FT, E)   # time -> frequency

    Fourier.hilbert!(E, FT, S)   # spectrum real to signal analytic

    rho = CuArrays.zeros(T, (grid.Nr, grid.Nt))
    kdrho = CuArrays.zeros(T, (grid.Nr, grid.Nt))

    HT = HankelTransforms.plan(grid.rmax, S)
    return FieldRT(w0, E, S, rho, kdrho, HT, FT)
end


function Field(
    unit::Units.UnitXY,
    grid::Grids.GridXY,
    lam0::T,
    initial_condition::Function,
) where T<:AbstractFloat
    w0 = 2 * pi * C0 / lam0
    w0 = convert(T, w0)

    E = initial_condition(grid.x, grid.y, unit.x, unit.y, unit.I)
    E = CuArrays.CuArray{Complex{T}}(E)

    FT = Fourier.FourierTransformXY(grid.Nx, grid.Ny)
    return FieldXY(w0, E, FT)
end


end

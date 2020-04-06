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


function peak_intensity(field::Field)
    return Float64(maximum(abs2.(field.E)))
end


"""
Total energy:
    W = 2 * pi * Int[|E(r, t)|^2 * r * dr * dt],   [W] = J
"""
function energy(grid::Grids.GridRT, field::FieldRT)
    return sum(abs2.(field.E) .* grid.rdr) * 2 * pi * grid.dt
end


function energy_gauss(grid::Grids.GridRT, field::FieldRT)
    t0 = pulse_duration(grid, field)
    a0 = beam_radius(grid, field)
    I0 = peak_intensity(field)
    return pi^1.5 * t0 * a0^2 * I0
end


function energy_photon(field::Field)
    return HBAR * field.w0
end


"""
Fluence:
    F(r) = Int[|E(r, t)|^2 * dt],   [F(r)] = J/cm^2
"""
function fluence(grid::Grids.GridRT, field::FieldRT)
    F = sum(abs2.(field.E) .* FloatGPU(grid.dt), dims=2)
    return CuArrays.collect(F)[:, 1]
end


function peak_fluence(grid::Grids.GridT, field::FieldT)
    return sum(abs2.(field.E)) * grid.dt
end


function peak_fluence(grid::Grids.GridRT, field::FieldRT)
    return maximum(sum(abs2.(field.E), dims=2)) * grid.dt
end


function peak_fluence_gauss(
    grid::Union{Grids.GridT, Grids.GridRT},
    field::Union{FieldT, FieldRT},
)
    t0 = pulse_duration(grid, field)
    I0 = peak_intensity(field)
    return sqrt(pi) * t0 * I0
end


function beam_radius(grid::Grids.GridR, field::FieldR)
    I = abs2.(field.E)
    I = CuArrays.collect(I)
    # Factor 2. because I(r) is only half of full distribution I(x):
    return 2 * Grids.radius(grid.r, I)
end


function beam_radius(grid::Grids.GridRT, field::FieldRT)
    F = fluence(grid, field)
    # Factor 2. because F(r) is only half of full distribution F(x):
    return 2 * Grids.radius(grid.r, F)
end


function beam_radius(grid::Grids.GridXY, field::FieldXY)
    I = abs2.(field.E)
    I = CuArrays.collect(I)
    Imax, imax = findmax(I)
    ax = Grids.radius(grid.x, I[:, imax[2]])
    ay = Grids.radius(grid.y, I[imax[1], :])
    return sqrt(ax * ay)
end


"""
Temporal fluence:
    F(t) = 2 * pi * Int[|E(r, t)|^2 * r * dr],   [F(t)] = W
"""
function temporal_fluence(grid::Grids.GridRT, field::FieldRT)
    F = sum(abs2.(field.E) .* grid.rdr .* FloatGPU(2 * pi), dims=1)
    return CuArrays.collect(F)[1, :]
end


function pulse_duration(grid::Grids.GridT, field::FieldT)
    I = abs2.(field.E)
    return Grids.radius(grid.t, I)
end


function pulse_duration(grid::Grids.GridRT, field::FieldRT)
    F = temporal_fluence(grid, field)
    return Grids.radius(grid.t, F)
end


function peak_power(grid::Grids.GridR, field::FieldR)
    return sum(abs2.(field.E) .* grid.rdr) * 2 * pi
end


function peak_power(grid::Grids.GridRT, field::FieldRT)
    W = energy(grid, field)
    t0 = pulse_duration(grid, field)
    return W / t0 / sqrt(pi)
end


function peak_power(grid::Grids.GridXY, field::FieldXY)
    return sum(abs2.(field.E)) * grid.dx * grid.dy
end


function peak_power_gauss(grid::Grids.Grid, field::Field)
    a0 = beam_radius(grid, field)
    I0 = peak_intensity(field)
    return pi * a0^2 * I0
end


end

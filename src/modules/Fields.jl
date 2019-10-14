module Fields

import CuArrays
import FFTW

import Fourier
import Grids
import Units

import PyCall
scipy_constants = PyCall.pyimport("scipy.constants")
const C0 = scipy_constants.c   # speed of light in vacuum
const HBAR = scipy_constants.hbar   # the Planck constant (divided by 2*pi) [J*s]

const FloatGPU = Float32
const ComplexGPU = ComplexF32


abstract type Field end


struct FieldR <: Field
    lam0 :: Float64
    f0 :: Float64
    w0 :: Float64
    E :: CuArrays.CuArray{ComplexGPU, 1}
end


struct FieldT{T<:AbstractFloat} <: Field
    lam0 :: T
    f0 :: T
    w0 :: T
    E :: AbstractArray{Complex{T}, 1}
    S :: AbstractArray{Complex{T}, 1}
    rho :: AbstractArray{T, 1}
    Kdrho :: AbstractArray{T, 1}
end


struct FieldRT <: Field
    lam0 :: Float64
    f0 :: Float64
    w0 :: Float64
    E :: CuArrays.CuArray{ComplexGPU, 2}
    S :: CuArrays.CuArray{ComplexGPU, 2}
    rho :: CuArrays.CuArray{FloatGPU, 2}
    Kdrho :: CuArrays.CuArray{FloatGPU, 2}
end


struct FieldXY <: Field
    lam0 :: Float64
    f0 :: Float64
    w0 :: Float64
    E :: CuArrays.CuArray{ComplexGPU, 2}
end


function Field(unit::Units.UnitR, grid::Grids.GridR, p::Tuple)
    lam0, initial_condition = p

    f0 = C0 / lam0
    w0 = 2. * pi * f0

    E = initial_condition(grid.r, unit.r, unit.I)
    E = CuArrays.CuArray(convert(Array{ComplexGPU, 1}, E))
    return FieldR(lam0, f0, w0, E)
end


function Field(unit::Units.UnitT, grid::Grids.GridT, p::Tuple)
    lam0, initial_condition = p

    f0 = C0 / lam0
    w0 = 2. * pi * f0

    E = initial_condition(grid.t, unit.t, unit.I)
    E = convert(Array{ComplexF64, 1}, E)

    S = zeros(ComplexF64, grid.Nw)
    Fourier.rfft!(grid.FT, E, S)   # time -> frequency

    Fourier.hilbert!(grid.FT, S, E)   # spectrum real to signal analytic

    rho = zeros(grid.Nt)
    Kdrho = zeros(grid.Nt)

    # Initialize a dummy GPU array in order to trigger the creation of the
    # device context. This will allow to call CUDAdrv.synchronize() in the
    # main cycle.
    tmp = CuArrays.zeros(1)

    return FieldT(lam0, f0, w0, E, S, rho, Kdrho)
end


function Field(unit::Units.UnitRT, grid::Grids.GridRT, p::Tuple)
    lam0, initial_condition = p

    f0 = C0 / lam0
    w0 = 2. * pi * f0

    E = initial_condition(grid.r, grid.t, unit.r, unit.t, unit.I)
    E = CuArrays.CuArray(convert(Array{ComplexGPU, 2}, E))

    S = CuArrays.zeros(ComplexGPU, (grid.Nr, grid.Nw))
    Fourier.rfft2!(grid.FT, E, S)   # time -> frequency

    Fourier.hilbert2!(grid.FT, S, E)   # spectrum real to signal analytic

    rho = CuArrays.zeros(FloatGPU, (grid.Nr, grid.Nt))
    Kdrho = CuArrays.zeros(FloatGPU, (grid.Nr, grid.Nt))

    return FieldRT(lam0, f0, w0, E, S, rho, Kdrho)
end


function Field(unit::Units.UnitXY, grid::Grids.GridXY, p::Tuple)
    lam0, initial_condition = p

    f0 = C0 / lam0
    w0 = 2. * pi * f0

    E = initial_condition(grid.x, grid.y, unit.x, unit.y, unit.I)
    E = CuArrays.CuArray(convert(Array{ComplexGPU, 2}, E))
    return FieldXY(lam0, f0, w0, E)
end


function peak_intensity(field::Field)
    return Float64(maximum(abs2.(field.E)))
end


"""
Total energy:
    W = 2 * pi * Int[|E(r, t)|^2 * r * dr * dt],   [W] = J
"""
function energy(grid::Grids.GridRT, field::FieldRT)
    return sum(abs2.(field.E) .* grid.rdr) * 2. * pi * grid.dt
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
    return 2. * Grids.radius(grid.r, I)
end


function beam_radius(grid::Grids.GridRT, field::FieldRT)
    F = fluence(grid, field)
    # Factor 2. because F(r) is only half of full distribution F(x):
    return 2. * Grids.radius(grid.r, F)
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
    F = sum(abs2.(field.E) .* grid.rdr .* FloatGPU(2. * pi), dims=1)
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
    return sum(abs2.(field.E) .* grid.rdr) * 2. * pi
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


"""
Integral power spectrum:
    Ew = rfft(Et)
    Ew = 2. * Ew * dt
    S = 2 * pi * Int[|Ew|^2 * r * dr]
"""
function integral_power_spectrum(grid::Grids.GridRT, field::FieldRT)
    S = sum(abs2.(field.S) .* grid.rdr .* FloatGPU(8. * pi * grid.dt^2), dims=1)
    return CuArrays.collect(S)[1, :]
end


end

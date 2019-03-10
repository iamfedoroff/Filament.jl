module Fields

import FFTW
import CuArrays

import PyCall

import Units
import Grids
import Fourier
import FourierGPU

const FloatGPU = Float32
const ComplexGPU = ComplexF32

scipy_constants = PyCall.pyimport("scipy.constants")
const C0 = scipy_constants.c   # speed of light in vacuum
const HBAR = scipy_constants.hbar   # the Planck constant (divided by 2*pi) [J*s]


struct Field
    lam0 :: Float64
    f0 :: Float64
    w0 :: Float64

    E :: CuArrays.CuArray{ComplexGPU, 2}
    S :: CuArrays.CuArray{ComplexGPU, 2}
end


function Field(unit::Units.Unit, grid::Grids.Grid, lam0::Float64,
               initial_condition::Function)
    f0 = C0 / lam0
    w0 = 2. * pi * f0

    E = initial_condition(grid.r, grid.t, unit.r, unit.t, unit.I)
    for i=1:grid.Nr
        E[i, :] = Fourier.signal_real_to_signal_analytic(grid.FT, real(E[i, :]))
    end
    E = CuArrays.CuArray(convert(Array{ComplexGPU, 2}, E))

    S = CuArrays.cuzeros(ComplexGPU, (grid.Nr, grid.Nw))
    FourierGPU.rfft2!(grid.FTGPU, E, S)

    return Field(lam0, f0, w0, E, S)
end


function peak_intensity(field::Field)
    return Float64(maximum(abs2.(field.E)))
end


"""
Total energy:
    W = 2 * pi * Int[|E(r, t)|^2 * r * dr * dt],   [W] = J
"""
function energy(grid::Grids.Grid, field::Field)
    return sum(abs2.(field.E) .* grid.rdr) * 2. * pi * grid.dt
end


function energy_gauss(grid::Grids.Grid, field::Field)
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
function fluence(grid::Grids.Grid, field::Field)
    F = sum(abs2.(field.E) .* FloatGPU(grid.dt), dims=2)
    return CuArrays.collect(F)[:, 1]
end


function peak_fluence(grid::Grids.Grid, field::Field)
    return maximum(sum(abs2.(field.E), dims=2)) * grid.dt
end


function peak_fluence_gauss(grid::Grids.Grid, field::Field)
    t0 = pulse_duration(grid, field)
    I0 = peak_intensity(field)
    return sqrt(pi) * t0 * I0
end


function beam_radius(grid::Grids.Grid, field::Field)
    F = fluence(grid, field)
    # Factor 2. because F(r) is only half of full distribution F(x):
    return 2. * Grids.radius(grid.r, F)
end


"""
Temporal fluence:
    F(t) = 2 * pi * Int[|E(r, t)|^2 * r * dr],   [F(t)] = W
"""
function temporal_fluence(grid::Grids.Grid, field::Field)
    F = sum(abs2.(field.E) .* grid.rdr .* FloatGPU(2. * pi), dims=1)
    return CuArrays.collect(F)[1, :]
end


function pulse_duration(grid::Grids.Grid, field::Field)
    F = temporal_fluence(grid, field)
    return Grids.radius(grid.t, F)
end


function peak_power(grid::Grids.Grid, field::Field)
    W = energy(grid, field)
    t0 = pulse_duration(grid, field)
    return W / t0 / sqrt(pi)
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
function integral_power_spectrum(grid::Grids.Grid, field::Field)
    S = sum(abs2.(field.S) .* grid.rdr .* FloatGPU(8. * pi * grid.dt^2), dims=1)
    return CuArrays.collect(S)[1, :]
end


end

module Fields

import FFTW
import CuArrays

using PyCall
@pyimport scipy.constants as sc

import Units
import Grids
import Fourier
import FourierGPU

const FloatGPU = Float32
const ComplexGPU = ComplexF32

const C0 = sc.c   # speed of light in vacuum
const HBAR = sc.hbar   # the Planck constant (divided by 2*pi) [J*s]


struct Field
    lam0 :: Float64
    f0 :: Float64
    w0 :: Float64

    E :: Array{ComplexF64, 2}
    E_gpu :: CuArrays.CuArray{ComplexGPU, 2}
    S_gpu :: CuArrays.CuArray{ComplexGPU, 2}
end


function Field(unit::Units.Unit, grid::Grids.Grid, lam0::Float64,
               initial_condition::Function)
    f0 = C0 / lam0
    w0 = 2. * pi * f0

    E = initial_condition(grid.r, grid.t, unit.r, unit.t, unit.I)
    for i=1:grid.Nr
        E[i, :] = Fourier.signal_real_to_signal_analytic(grid.FT, real(E[i, :]))
    end

    E_gpu = CuArrays.cu(convert(Array{ComplexGPU, 2}, E))

    S_gpu = CuArrays.cuzeros(ComplexGPU, (grid.Nr, grid.Nw))
    FourierGPU.rfft2!(grid.FTGPU, E_gpu, S_gpu)

    return Field(lam0, f0, w0, E, E_gpu, S_gpu)
end


function peak_intensity(field::Field)
    return Float64(maximum(abs2.(field.E_gpu)))
end


"""
Total energy:
    W = 2 * pi * Int[|E(r, t)|^2 * r * dr * dt],   [W] = J
"""
function energy(grid::Grids.Grid, field::Field)
    return sum(abs2.(field.E_gpu) .* grid.r_gpu .* grid.dr_gpu) * 2. * pi * grid.dt
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
    F = sum(abs2.(field.E_gpu) .* FloatGPU(grid.dt), dims=2)
    return convert(Array{Float64, 1}, CuArrays.collect(F)[:, 1])
end


function peak_fluence(grid::Grids.Grid, field::Field)
    return maximum(sum(abs2.(field.E_gpu), dims=2)) * grid.dt
end


function peak_fluence_gauss(grid::Grids.Grid, field::Field)
    t0 = pulse_duration(grid, field)
    I0 = peak_intensity(field)
    return sqrt(pi) * t0 * I0
end


function beam_radius(grid::Grids.Grid, field::Field)
    F = fluence(grid, field)
    # Factor 2. because F(r) is only half of full distribution F(x):
    return 2. * radius(grid.r, F)
end


"""
Temporal fluence:
    F(t) = 2 * pi * Int[|E(r, t)|^2 * r * dr],   [F(t)] = W
"""
function temporal_fluence(grid::Grids.Grid, field::Field)
    F = sum(abs2.(field.E_gpu) .* grid.r_gpu .* grid.dr_gpu .*
            FloatGPU(2. * pi), dims=1)
    return convert(Array{Float64, 1}, CuArrays.collect(F)[1, :])
end


function pulse_duration(grid::Grids.Grid, field::Field)
    F = temporal_fluence(grid, field)
    return radius(grid.t, F)
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
    S = sum(abs2.(field.S_gpu) .* grid.r_gpu .* grid.dr_gpu .*
            FloatGPU(8. * pi * grid.dt^2), dims=1)
    return convert(Array{Float64, 1}, CuArrays.collect(S)[1, :])
end


function radius(x::Array{Float64, 1}, y::Array{Float64, 1},
                level::Float64=exp(-1.))
    Nx = length(x)
    ylevel = maximum(y) * level

    radl = 0.
    for i=1:Nx
        if y[i] >= ylevel
            radl = x[i]
            break
        end
    end

    radr = 0.
    for i=Nx:-1:1
        if y[i] >= ylevel
            radr = x[i]
            break
        end
    end

    return 0.5 * (abs(radl) + abs(radr))
end


end

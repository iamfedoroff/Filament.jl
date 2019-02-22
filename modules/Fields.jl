module Fields

import FFTW
import CuArrays

using PyCall
@pyimport scipy.constants as sc

import Units
import Grids
import Fourier

const ComplexGPU = ComplexF32

const C0 = sc.c   # speed of light in vacuum
const HBAR = sc.hbar   # the Planck constant (divided by 2*pi) [J*s]


mutable struct Field
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

    return Field(lam0, f0, w0, E, E_gpu, S_gpu)
end


function peak_intensity(field::Field)
    return maximum(abs2.(field.E))
end


"""
Total energy:
    W = Int[|E(r,t)|^2 * 2*pi*r*dr*dt],   [W] = J
"""
function energy(grid::Grids.Grid, field::Field)
    W = 0.
    for j=1:grid.Nt
        for i=1:grid.Nr
            W = W + abs2(field.E[i, j]) * grid.r[i] * grid.dr[i]
        end
    end
    return W * 2. * pi * grid.dt
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
    F(r) = Int[|E(r,t)|^2 * dt],   [F(r)] = J/cm^2
"""
function fluence(grid::Grids.Grid, field::Field)
    F = zeros(grid.Nr)
    for j=1:grid.Nt
        for i=1:grid.Nr
            F[i] = F[i] + abs2(field.E[i, j])
        end
    end
    return F * grid.dt
end


function peak_fluence(grid::Grids.Grid, field::Field)
    F = fluence(grid, field)
    return maximum(F)
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
    F(t) = Int[|E(r,t)|^2 * 2*pi*r*dr],   [F(t)] = W
"""
function temporal_fluence(grid::Grids.Grid, field::Field)
    F = zeros(grid.Nt)
    for j=1:grid.Nt
        for i=1:grid.Nr
            F[j] = F[j] + abs2(field.E[i, j]) * grid.r[i] * grid.dr[i]
        end
    end
    return F * 2. * pi
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
    Ew = FFT[Et]
    S = Int[|Ew|^2 * 2*pi*r*dr]
"""
function integral_power_spectrum(grid::Grids.Grid, field::Field)
    S = zeros(grid.Nw)
    for i=1:grid.Nr
        Et = real(field.E[i, :])
        Ew = FFTW.rfft(Et)
        Ew = 2. * Ew * grid.dt
        S = S + abs2.(Ew) * grid.r[i] * grid.dr[i]
    end
    return S * 2. * pi
end


function radius(x::Array{Float64, 1}, y::Array{Float64, 1},
                level::Float64=1. / exp(1.))
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

    rad = 0.5 * (abs(radl) + abs(radr))
    return rad
end


end

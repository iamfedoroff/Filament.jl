module Fields

using PyCall
@pyimport scipy.constants as sc

import Grids
import Fourier

const C0 = sc.c   # speed of light in vacuum
const HBAR = sc.hbar   # the Planck constant (divided by 2*pi) [J*s]


mutable struct Field
    lam0 :: Float64
    f0 :: Float64
    w0 :: Float64

    grid :: Grids.Grid

    E :: Array{Complex128, 2}
    S :: Array{Complex128, 2}
    rho :: Array{Float64, 1}
end


function Field(unit, grid, lam0, initial_condition::Function)
    f0 = C0 / lam0
    w0 = 2. * pi * f0

    E = initial_condition(grid.r, grid.t, unit.r, unit.t, unit.I)
    for i=1:grid.Nr
        E[i, :] = Fourier.signal_real_to_analytic(real(E[i, :]))
    end

    S = zeros(Complex128, (grid.Nr, grid.Nw))

    rho = zeros(grid.Nr)

    return Field(lam0, f0, w0, grid, E, S, rho)
end


function peak_intensity(field)
    Imax = maximum(abs2.(field.E))
    return Imax
end


"""
Total energy:
    W = Int[|E(r,t)|^2 * 2*pi*r*dr*dt],   [W] = J
"""
function energy(field)
    r = field.grid.r
    W = 0.
    for j=1:field.grid.Nt
        for i=1:field.grid.Nr
            dr = step(i, r)
            W = W + abs2(field.E[i, j]) * r[i] * dr
        end
    end
    W = W * 2. * pi * field.grid.dt
    return W
end


function energy_gauss(field)
    t0 = pulse_duration(field)
    a0 = beam_radius(field)
    I0 = peak_intensity(field)
    Wg = pi^1.5 * t0 * a0^2 * I0
    return Wg
end


function energy_photon(field)
    Wph = HBAR * field.w0
    return Wph
end


"""
Fluence:
    F(r) = Int[|E(r,t)|^2 * dt],   [F(r)] = J/cm^2
"""
function fluence(field)
    F = zeros(field.grid.Nr)
    for j=1:field.grid.Nt
        for i=1:field.grid.Nr
            F[i] = F[i] + abs2(field.E[i, j])
        end
    end
    F = F * field.grid.dt
    return F
end


function peak_fluence(field)
    F = fluence(field)
    Fmax = maximum(F)
    return Fmax
end


function peak_fluence_gauss(field)
    t0 = pulse_duration(field)
    I0 = peak_intensity(field)
    Fmaxg = sqrt(pi) * t0 * I0
    return Fmaxg
end


function beam_radius(field)
    F = fluence(field)
    a0 = 2. * radius(field.grid.r, F)
    # Factor 2. because F(r) is only half of full distribution F(x)
    return a0
end


"""
Temporal fluence:
    F(t) = Int[|E(r,t)|^2 * 2*pi*r*dr],   [F(t)] = W
"""
function temporal_fluence(field)
    r = field.grid.r
    F = zeros(field.grid.Nt)
    for j=1:field.grid.Nt
        for i=1:field.grid.Nr
            dr = step(i, r)
            F[j] = F[j] + abs2(field.E[i, j]) * r[i] * dr
        end
    end
    F = F * 2. * pi
    return F
end


function pulse_duration(field)
    F = temporal_fluence(field)
    t0 = radius(field.grid.t, F)
    return t0
end


function peak_power(field)
    W = energy(field)
    t0 = pulse_duration(field)
    P = W / t0 / sqrt(pi)
    return P
end


function peak_power_gauss(field)
    a0 = beam_radius(field)
    I0 = peak_intensity(field)
    Pg = pi * a0^2 * I0
    return Pg
end


function peak_plasma_density(field)
    rhomax = maximum(field.rho)
    return rhomax
end


function plasma_radius(field)
    rad = 2. * radius(field.grid.r, field.rho)
    # Factor 2. because rho(r) is only half of full distribution rho(x)
    return rad
end


"""
Linear plasma density:
    lrho = Int[Ne * 2*pi*r*dr],   [De] = 1/m
"""
function linear_plasma_density(field)
    r = field.grid.r
    lrho = 0.
    for i=1:field.grid.Nr
        dr = step(i, r)
        lrho = lrho + field.rho[i] * r[i] * dr
    end
    lrho = lrho * 2. * pi
    return lrho
end


"""
Integral power spectrum:
    Ew = FFT[Et]
    S = Int[|Ew|^2 * 2*pi*r*dr]
"""
function integral_power_spectrum(field)
    r = field.grid.r
    S = zeros(field.grid.Nw)
    for i=1:field.grid.Nr
        Et = real(field.E[i, :])
        Ew = Fourier.rfft1d(Et)
        Ew = 2. * Ew * field.grid.dt

        dr = step(i, r)
        S = S + abs2.(Ew) * r[i] * dr
    end
    S = S * 2. * pi
    return S
end


"""Step dx at a specific point i of a nonuniform grid x"""
function step(i, x)
    Nx = length(x)
    if i == 1
        dx = x[2] - x[1]
    elseif i == Nx
        dx = x[Nx] - x[Nx - 1]
    else
        dx = 0.5 * (x[i+1] - x[i-1])
    end
    return dx
end


function radius(x, y, level=1. / exp(1.))
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

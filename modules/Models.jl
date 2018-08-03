module Models

import CuArrays
import CUDAnative
using PyCall
@pyimport scipy.constants as sc
@pyimport matplotlib.pyplot as plt

import Units
import Grids
import Fields
import Media
import Plasmas
import Hankel
import HankelGPU
import Fourier
import FourierGPU
import RungeKuttas
import Guards

const C0 = sc.c   # speed of light in vacuum
const EPS0 = sc.epsilon_0   # the electric constant (vacuum permittivity) [F/m]
const MU0 = sc.mu_0   # the magnetic constant [N/A^2]
const QE = sc.e   # elementary charge [C]
const ME = sc.m_e   # electron mass [kg]
const HBAR = sc.hbar   # the Planck constant (divided by 2*pi) [J*s]

const FloatGPU = Float32
const ComplexGPU = Complex64


struct Model
    KZ_gpu :: CuArrays.CuArray{ComplexGPU, 2}
    QZ :: Array{Complex128, 2}
    Rk :: Float64
    Rr :: Float64
    Hramanw :: Array{Complex128, 1}
    Rp :: Array{Complex128, 1}
    Ra :: Array{Complex128, 1}
    phi_kerr :: Float64
    phi_plasma :: Float64
    guard :: Guards.GuardFilter
    RK :: Union{RungeKuttas.RungeKutta2, RungeKuttas.RungeKutta3,
                RungeKuttas.RungeKutta4}
    keys :: Dict{String, Any}
end


function Model(unit::Units.Unit, grid::Grids.Grid, field::Fields.Field,
               medium::Media.Medium, plasma::Plasmas.Plasma,
               keys::Dict{String, Any})
    # Guards -------------------------------------------------------------------
    rguard_width = keys["rguard_width"]
    tguard_width = keys["tguard_width"]
    kguard = keys["kguard"]
    wguard = keys["wguard"]
    guard = Guards.GuardFilter(unit, grid, medium,
                               rguard_width, tguard_width, kguard, wguard)

    # Runge-Kutta --------------------------------------------------------------
    RKORDER = keys["RKORDER"]
    RK = RungeKuttas.RungeKutta(RKORDER, grid.Nr, grid.Nw)

    # Linear propagator --------------------------------------------------------
    KPARAXIAL = keys["KPARAXIAL"]

    beta = Media.beta_func.(medium, grid.w * unit.w)
    KZ = zeros(Complex128, (grid.Nr, grid.Nw))
    if KPARAXIAL != 0
        for j=1:grid.Nw
            if beta[j] != 0.
                for i=1:grid.Nr
                    KZ[i, j] = beta[j] - (grid.k[i] * unit.k)^2 / (2. * beta[j])
                end
            end
        end
    else
        for j=1:grid.Nw
            for i=1:grid.Nr
                KZ[i, j] = sqrt(beta[j]^2 - (grid.k[i] * unit.k)^2 + 0im)
            end
        end
    end

    vf = Media.group_velocity(medium, field.w0)   # frame velocity
    for j=1:grid.Nw
        for i=1:grid.Nr
            KZ[i, j] = (KZ[i, j] - grid.w[j] * unit.w / vf) * unit.z
        end
    end

    @. KZ = conj(KZ)

    KZ_gpu = CuArrays.CuArray(convert(Array{ComplexGPU, 2}, KZ))

    # Nonlinear propagator -----------------------------------------------------
    QPARAXIAL = keys["QPARAXIAL"]

    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    mu = medium.permeability(grid.w * unit.w)

    Qfactor = @. MU0 * mu * (grid.w * unit.w)^2 / 2. * unit.z / Eu

    QZ = zeros(Complex128, (grid.Nr, grid.Nw))
    if QPARAXIAL != 0
        for j=1:grid.Nw
            if beta[j] != 0.
                for i=1:grid.Nr
                    QZ[i, j] = Qfactor[j] / beta[j]
                end
            end
        end
    else
        for j=1:grid.Nw
            for i=1:grid.Nr
                kzij = sqrt(beta[j]^2 - (grid.k[i] * unit.k)^2 + 0im)
                if kzij != 0.
                    QZ[i, j] = Qfactor[j] / kzij
                end
            end
        end
    end

    @. QZ = conj(QZ)

    # Kerr nonlinearity --------------------------------------------------------
    Rk = Rk_func(unit, field, medium)
    phi_kerr = phi_kerr_func(unit, field, medium)

    # Stimulated Raman nonlinearity --------------------------------------------
    RAMAN = keys["RAMAN"]

    graman = medium.graman

    if RAMAN != 0
        Rk = (1. - graman) * Rk
        Rr = graman * Rk
    else
        Rr = 0.
    end

    # For assymetric grids, where abs(tmin) != tmax, we need tshift to put H(t)
    # into the grid center (see "circular convolution"):
    tshift = grid.tmin + 0.5 * (grid.tmax - grid.tmin)
    Hraman = @. medium.raman_response((grid.t - tshift) * unit.t)
    Hraman = Hraman * grid.dt * unit.t

    if abs(1. - sum(Hraman)) > 1e-6
        print("WARNING: The integral of Raman response function should be" *
              " normalized to 1.\n")
    end

    @. Hraman = Hraman * guard.T   # temporal filter
    Hramanw = Fourier.rfft1d(grid.FT, Hraman)   # time -> frequency

    # Plasma nonlinearity ------------------------------------------------------
    Rp = Rp_func(unit, grid, field, medium, plasma)
    @. Rp = conj(Rp)
    phi_plasma = phi_kerr_func(unit, field, medium)

    # Losses due to multiphoton ionization -------------------------------------
    Ra = Ra_func(unit, grid, field, medium)
    @. Ra = conj(Ra)

    return Model(KZ_gpu, QZ, Rk, Rr, Hramanw, Rp, Ra, phi_kerr, phi_plasma,
                 guard, RK, keys)
end


function adaptive_dz(model::Model, AdaptLevel::Float64, I::Float64,
                     rho::Float64)
    if model.keys["KERR"] != 0
        dz_kerr = model.phi_kerr / I * AdaptLevel
    else
        dz_kerr = Inf
    end

    if (model.keys["PLASMA"] != 0) & (rho != 0.)
        dz_plasma = model.phi_plasma / rho * AdaptLevel
    else
        dz_plasma = Inf
    end

    dz = min(dz_kerr, dz_plasma)
    return dz
end


function zstep(dz::Float64, grid::Grids.Grid, field::Fields.Field,
               plasma::Plasmas.Plasma, model::Models.Model)


    function func(S::Array{Complex128, 2})
        res = zeros(Complex128, (grid.Nr, grid.Nw))
        Ea = zeros(Complex128, grid.Nt)
        Sa = zeros(Complex128, grid.Nt)
        Et = zeros(Float64, grid.Nt)
        St = zeros(Complex128, grid.Nw)
        Ftmp = zeros(Float64, grid.Nt)
        Stmp = zeros(Complex128, grid.Nw)
        Iconv = zeros(Float64, grid.Nt)
        rhot = zeros(Float64, grid.Nt)
        Kdrhot = zeros(Float64, grid.Nt)

        for i=1:grid.Nr
            @inbounds @views @. St = S[i, :]
            Fourier.spectrum_real_to_signal_analytic!(grid.FT, St, Ea)
            @inbounds @. Et = real(Ea)

            # Kerr nonlinearity:
            if model.keys["KERR"] != 0
                if model.keys["THG"] != 0
                    @inbounds @. Ftmp = Et^3
                else
                    @inbounds @. Ftmp = 3. / 4. * abs2(Ea) * Et
                end
                @inbounds @. Ftmp = Ftmp * model.guard.T   # temporal filter
                Fourier.rfft1d!(grid.FT, Ftmp, Stmp)   # time -> frequency

                @inbounds @views @. res[i, :] = res[i, :] + model.Rk * Stmp
            end

            # Stimulated Raman nonlinearity:
            if model.keys["RAMAN"] != 0
                if model.keys["RTHG"] != 0
                    @inbounds @. Ftmp = Et^2
                else
                    @inbounds @. Ftmp = 3. / 4. * abs2(Ea)
                end
                @inbounds @. Ftmp = Ftmp * model.guard.T   # temporal filter
                Fourier.rfft1d!(grid.FT, Ftmp, Stmp)   # time -> frequency
                @inbounds @. Stmp = Stmp * model.Hramanw
                Fourier.irfft1d!(grid.FT, Stmp, Iconv)   # frequency -> time
                Iconv = Fourier.roll(Iconv, div(grid.Nt + 1, 2) + 1)

                @inbounds @. Ftmp = Iconv * Et
                @inbounds @. Ftmp = Ftmp * model.guard.T   # temporal filter
                Fourier.rfft1d!(grid.FT, Ftmp, Stmp)   # time -> frequency

                @inbounds @views @. res[i, :] = res[i, :] + model.Rr * Stmp
            end

            # Plasma nonlinearity:
            if model.keys["PLASMA"] != 0
                @inbounds @views @. rhot = plasma.rho[i, :]
                @inbounds @. Ftmp = rhot * Et
                @inbounds @. Ftmp = Ftmp * model.guard.T   # temporal filter
                Fourier.rfft1d!(grid.FT, Ftmp, Stmp)   # time -> frequency

                @inbounds @views @. res[i, :] = res[i, :] + model.Rp * Stmp
            end

            # Losses due to multiphoton ionization:
            if model.keys["ILOSSES"] != 0
                @inbounds @views @. Kdrhot = plasma.Kdrho[i, :]

                if model.keys["IONARG"] != 0
                    @inbounds @. Ftmp = abs2(Ea)
                else
                    @inbounds @. Ftmp = Et^2
                end

                @inbounds for j=1:grid.Nt
                    if Ftmp[j] >= 1e-30
                        Ftmp[j] = Kdrhot[j] / Ftmp[j] * Et[j]
                    else
                        Ftmp[j] = 0.
                    end
                end
                @inbounds @. Ftmp = Ftmp * model.guard.T   # temporal filter
                Fourier.rfft1d!(grid.FT, Ftmp, Stmp)   # time -> frequency

                @inbounds @views @. res[i, :] = res[i, :] + model.Ra * Stmp
            end
        end

        # Nonparaxiality:
        if model.keys["QPARAXIAL"] != 0
            @inbounds @. res = -1im * model.QZ * res
        else
            for j=1:grid.Nw
                res[:, j] = Hankel.dht(grid.HT, res[:, j])
            end
            @inbounds @. res = -1im * model.QZ * res
            @inbounds @. res = res * model.guard.K   # angular filter
            for j=1:grid.Nw
                res[:, j] = Hankel.idht(grid.HT, res[:, j])
            end
        end

        return res
    end


    # Calculate plasma density -------------------------------------------------
    if (model.keys["PLASMA"] != 0) | (model.keys["ILOSSES"] != 0)
        Plasmas.free_charge(plasma, grid, field)
    end

    # dz_gpu = convert(FloatGPU, dz)
    # E_gpu = CuArrays.CuArray(convert(Array{ComplexGPU, 2}, field.E))
    # S_gpu = CuArrays.CuArray(convert(Array{ComplexGPU, 2}, field.S))

    # Field -> temporal spectrum -----------------------------------------------
    # FourierGPU.rfft2d!(grid.FTGPU, E_gpu, S_gpu)
    Fourier.rfft2d!(grid.FT, field.E, field.S)

    # Nonlinear propagator -----------------------------------------------------
    if (model.keys["KERR"] != 0) | (model.keys["PLASMA"] != 0) |
       (model.keys["ILOSSES"] != 0)
        RungeKuttas.RungeKutta_calc!(model.RK, field.S, dz, func)
    end

    # Linear propagator --------------------------------------------------------
    dz_gpu = convert(FloatGPU, dz)
    E_gpu = CuArrays.CuArray(convert(Array{ComplexGPU, 2}, field.E))
    S_gpu = CuArrays.CuArray(convert(Array{ComplexGPU, 2}, field.S))

    HankelGPU.dht!(grid.HTGPU, S_gpu)
    @inbounds @. S_gpu = S_gpu * exp_cuda(model.KZ_gpu * dz_gpu)
    @inbounds @. S_gpu = S_gpu * model.guard.K_gpu   # angular filter
    HankelGPU.idht!(grid.HTGPU, S_gpu)

    # Temporal spectrum -> field -----------------------------------------------
    # spectral filter:
    for i=1:grid.Nr
        @inbounds S_gpu[i, :] = S_gpu[i, :] .* model.guard.W_gpu
    end

    FourierGPU.spectrum_real_to_signal_analytic_2d!(grid.FTGPU, S_gpu, E_gpu)

    # spatial filter:
    for j=1:grid.Nt
        @inbounds E_gpu[:, j] = E_gpu[:, j] .* model.guard.R_gpu
    end

    # temporal filter:
    for i=1:grid.Nr
        @inbounds E_gpu[i, :] = E_gpu[i, :] .* model.guard.T_gpu
    end

    field.E = convert(Array{Complex128, 2}, CuArrays.collect(E_gpu))

    return nothing
end


function Rk_func(unit::Units.Unit, field::Fields.Field, medium::Media.Medium)
    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    chi3 = Media.chi3_func(medium, field.w0)
    R = EPS0 * chi3 * Eu^3
    return R
end


function Rp_func(unit::Units.Unit, grid::Grids.Grid, field::Fields.Field,
                 medium::Media.Medium, plasma::Plasmas.Plasma)
    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    nuc = plasma.nuc
    MR = plasma.mr * ME   # reduced mass of electron and hole (effective mass)
    R = zeros(Complex128, grid.Nw)
    for i=1:grid.Nw
        if grid.w[i] != 0.
            R[i] = 1im / (grid.w[i] * unit.w) *
                   QE^2 / MR / (nuc - 1im * (grid.w[i] * unit.w)) *
                   unit.rho * Eu
        end
    end
    return R
end


function Ra_func(unit::Units.Unit, grid::Grids.Grid, field::Fields.Field,
                 medium::Media.Medium)
    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)

    R = zeros(Complex128, grid.Nw)
    for i=1:grid.Nw
        if grid.w[i] != 0.
            R[i] = 1im / (grid.w[i] * unit.w) *
                   HBAR * field.w0 * unit.rho / (unit.t * Eu)
        end
    end
    return R
end


"""Kerr phase factor for adaptive z step."""
function phi_kerr_func(unit::Units.Unit, field::Fields.Field,
                       medium::Media.Medium)
    w0 = field.w0
    n0 = real(Media.refractive_index(medium, field.w0))
    k0 = Media.k_func(medium, w0)
    Eu = Units.E(unit, n0)
    mu = medium.permeability(w0)
    chi3 = Media.chi3_func(medium, field.w0)
    Rk0 = mu * w0^2 / (2. * C0^2) * chi3 * Eu^2 * unit.z

    if real(Rk0) != 0.
        phi_real = k0 / (3. / 4. * abs(real(Rk0)))
    else
        phi_real = Inf
    end

    if imag(Rk0) != 0.
        phi_imag = k0 / (3. / 4. * abs(imag(Rk0)))
    else
        phi_imag = Inf
    end

    phi = min(phi_real, phi_imag)
    return phi
end


"""Plasma phase factor for adaptive z step."""
function phi_plasma(unit::Units.Unit, field::Fields.Field, medium::Media.Medium)
    w0 = field.w0
    k0 = Media.k_func(medium, w0)
    nuc = medium.nuc
    mu = medium.permeability(w0)
    MR = medium.mr * ME   # reduced mass of electron and hole (effective mass)
    Rp0 = 0.5 * MU0 * mu * w0 / (nuc - 1im * w0) * QE^2 / MR * unit.rho * unit.z

    if real(Rp0) != 0.
        phi_real = k0 / abs(real(Rp0))
    else
        phi_real = Inf
    end

    if imag(Rp0) != 0.
        phi_imag = k0 / abs(imag(Rp0))
    else
        phi_imag = Inf
    end

    phi = min(phi_real, phi_imag)
    return phi
end


"""
Calculates "exp(-1im * x)" on GPU.

Unfortunately, "CUDAnative.exp.(x)" function does not work with complex
arguments. To solve the issue, I use Euler's formula:
    exp(-1im * x) = (cos(xr) - 1im * sin(xr)) * exp(xi),
where xr = real(x) and xi = imag(x).
"""
function exp_cuda(x::ComplexGPU)
    xr = real(x)
    xi = imag(x)
    return (CUDAnative.cos(xr) - 1im * CUDAnative.sin(xr)) * CUDAnative.exp(xi)
end


end

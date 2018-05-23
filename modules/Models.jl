module Models

using PyCall
@pyimport scipy.constants as sc
@pyimport matplotlib.pyplot as plt

import Units
import Grids
import Fields
import Media
import Hankel
import Fourier

const C0 = sc.c   # speed of light in vacuum
const EPS0 = sc.epsilon_0   # the electric constant (vacuum permittivity) [F/m]
const MU0 = sc.mu_0   # the magnetic constant [N/A^2]
const QE = sc.e   # elementary charge [C]
const ME = sc.m_e   # electron mass [kg]
const HBAR = sc.hbar   # the Planck constant (divided by 2*pi) [J*s]


struct Model
    KZ :: Array{Complex128, 2}
    QZ :: Array{Complex128, 2}
    Rk :: Float64
    phi_kerr :: Float64
    Rguard :: Array{Float64, 1}
    Tguard :: Array{Float64, 1}
    Kguard :: Array{Float64, 2}
    Wguard :: Array{Float64, 1}
    keys :: Dict{String, Any}
end


function Model(unit::Units.Unit, grid::Grids.Grid, field::Fields.Field,
               medium::Media.Medium, keys::Dict{String, Any})
    # Guards -------------------------------------------------------------------
    # Spatial guard:
    rguard_width = keys["rguard_width"]
    Rguard = guard_window(grid.r, rguard_width, mode="right")

    # Temporal guard:
    tguard_width = keys["tguard_width"]
    Tguard = guard_window(grid.t, tguard_width, mode="both")

    # Angular filter:
    kguard = keys["kguard"]
    k = Media.k_func.(medium, grid.w * unit.w)
    kmax = k * sind(kguard)
    Kguard = zeros((grid.Nr, grid.Nw))
    for j=2:grid.Nw   # from 2 because kmax[1]=0 since w[1]=0
        for i=1:grid.Nr
            if kmax[j] != 0.
                Kguard[i, j] = exp(-((grid.k[i] * unit.k)^2 / kmax[j]^2)^20)
            end
        end
    end

    # Spectral filter:
    wguard = keys["wguard"]
    Wguard = @. exp(-((grid.w * unit.w)^2 / wguard^2)^20)

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

    # Kerr nonlinearity --------------------------------------------------------
    KERR = keys["KERR"]
    THG = keys["THG"]

    Rk = Rk_func(unit, medium, field)
    phi_kerr = phi_kerr_func(unit, medium, field)

    return Model(KZ, QZ,
                 Rk, phi_kerr,
                 Rguard, Tguard, Kguard, Wguard,
                 keys)
end


function adaptive_dz(model, AdaptLevel, I, rho)
    if model.keys["KERR"] != 0
        dz_kerr = model.phi_kerr / I * AdaptLevel
    else
        dz_kerr = Inf
    end

    # if (model.keys["PLASMA"] != 0) & (rho != 0.)
    #     dz_plasma = model.phi_plasma / rho * AdaptLevel
    # else
    #     dz_plasma = Inf
    # end
    dz_plasma = Inf

    dz = min(dz_kerr, dz_plasma)
    return dz
end


function zstep(dz::Float64, grid::Grids.Grid, field::Fields.Field,
               model::Models.Model)

    function func(S::Array{Complex128, 2})
        res = zeros(Complex128, (grid.Nr, grid.Nw))

        for i=1:grid.Nr
            Sa = Fourier.spectrum_real_to_analytic(S[i, :], grid.Nt)
            Ea = Fourier.ifft1d(Sa)   # frequency -> time
            Et = real(Ea)

            # Kerr nonlinearity:
            if model.keys["KERR"] != 0
                if model.keys["THG"] != 0
                    Ftmp = @. Et^3
                else
                    Ftmp = @. 3. / 4. * abs2(Ea) * Et
                end
                Ftmp = @. Ftmp * model.Tguard   # temporal filter
                Stmp = Fourier.rfft1d(Ftmp)   # time -> frequency

                res[i, :] = @. res[i, :] + model.Rk * Stmp
            end
        end

        # Nonparaxiality:
        if model.keys["QPARAXIAL"] != 0
            res = @. 1im * model.QZ * res
        else
            for j=1:grid.Nw
                res[:, j] = Hankel.dht(grid.HT, res[:, j])
            end
            res = @. 1im * model.QZ * res
            res = @. res * model.Kguard   # angular filter
            for j=1:grid.Nw
                res[:, j] = Hankel.idht(grid.HT, res[:, j])
            end
        end

        return res
    end


    # Field -> temporal spectrum -----------------------------------------------
    for i=1:grid.Nr
        field.S[i, :] = Fourier.rfft1d(real(field.E[i, :]))   # time -> frequency
    end

    # Nonlinear propagator -----------------------------------------------------
    if model.keys["KERR"] != 0
        # RK2:
        # k1 = dz * func(field.S)
        # k2 = dz * func(field.S + 2. / 3. * k1)
        # field.S = field.S + (k1 + 3. * k2) / 4.

        # RK3:
        k1 = dz * func(field.S)
        k2 = dz * func(field.S + 0.5 * k1)
        k3 = dz * func(field.S - k1 + 2. * k2)
        field.S = field.S + (k1 + 4. * k2 + k3) / 6.

        # RK4:
        # k1 = dz * func(field.S)
        # k2 = dz * func(field.S + 0.5 * k1)
        # k3 = dz * func(field.S + 0.5 * k2)
        # k4 = dz * func(field.S + k3)
        # field.S = field.S + (k1 + 2. * k2 + 2. * k3 + k4) / 6.
    end

    # Linear propagator --------------------------------------------------------
    for j=1:grid.Nw
        field.S[:, j] = Hankel.dht(grid.HT, field.S[:, j])
    end

    for j=1:grid.Nw
        for i=1:grid.Nr
            field.S[i, j] = field.S[i, j] * exp(1im * model.KZ[i, j] * dz)
            field.S[i, j] = field.S[i, j] * model.Kguard[i, j]   # angular filter
        end
    end

    for j=1:grid.Nw
        field.S[:, j] = Hankel.idht(grid.HT, field.S[:, j])
    end

    # Temporal spectrum -> field -----------------------------------------------
    for i=1:grid.Nr
        field.S[i, :] = @. field.S[i, :] * model.Wguard   # spectral filter
        Sa = Fourier.spectrum_real_to_analytic(field.S[i, :], grid.Nt)
        field.E[i, :] = Fourier.ifft1d(Sa)   # frequency -> time
        field.E[i, :] = @. field.E[i, :] * model.Rguard[i] * model.Tguard   # spatial and temporal filters
    end
end


function Rk_func(unit, medium, field)
    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    chi3 = Media.chi3_func(medium, field.w0)
    R = EPS0 * chi3 * Eu^3
    return R
end


"""Kerr phase factor for adaptive z step."""
function phi_kerr_func(unit, medium, field)
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


"""
Lossy guard window at the ends of grid coordinate.

    x: grid coordinate
    guard_width: the width of the lossy guard window
    mode: "left" - lossy guard only on the left end of the grid
          "right" - lossy guard only on the right end of the grid
          "both" - lossy guard on both ends of the grid
"""
function guard_window(x, guard_width; mode="both")
    assert(guard_width >= 0.)
    assert(mode in ["left", "right", "both"])

    if mode in ["left", "right"]
        assert(guard_width <= x[end] - x[1])
    else
        assert(guard_width <= 0.5 * (x[end] - x[1]))
    end

    Nx = length(x)

    if guard_width == 0
        guard = ones(Nx)
    else
        width = 0.5 * guard_width

        # Left guard
        guard_xmin = x[1]
        guard_xmax = x[2] + guard_width
        gauss1 = zeros(Nx)
        gauss2 = ones(Nx)
        for i=1:Nx
            if x[i] >= guard_xmin
                gauss1[i] = 1. - exp(-((x[i] - guard_xmin) / width)^6)
            end
            if x[i] <= guard_xmax
                gauss2[i] = exp(-((x[i] - guard_xmax) / width)^6)
            end
        end
        guard_left = 0.5 * (gauss1 + gauss2)

        # Right guard
        guard_xmin = x[end] - guard_width
        guard_xmax = x[end]
        gauss1 = ones(Nx)
        gauss2 = zeros(Nx)
        for i=1:Nx
            if x[i] >= guard_xmin
                gauss1[i] = exp(-((x[i] - guard_xmin) / width)^6)
            end
            if x[i] <= guard_xmax
                gauss2[i] = 1. - exp(-((x[i] - guard_xmax) / width)^6)
            end
        end
        guard_right = 0.5 * (gauss1 + gauss2)

        # Result guard:
        if mode == "left"
            guard = guard_left
        elseif mode == "right"
            guard = guard_right
        elseif mode == "both"
            guard = guard_left + guard_right - 1.
        end
    end
    return guard
end


end

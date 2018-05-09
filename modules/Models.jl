module Models

using PyCall
@pyimport scipy.constants as sc
@pyimport matplotlib.pyplot as plt

import Units
import Fields
import Media
import Hankel

C0 = sc.c   # speed of light in vacuum
EPS0 = sc.epsilon_0   # the electric constant (vacuum permittivity) [F/m]
MU0 = sc.mu_0   # the magnetic constant [N/A^2]
QE = sc.e   # elementary charge [C]
ME = sc.m_e   # electron mass [kg]
HBAR = sc.hbar   # the Planck constant (divided by 2*pi) [J*s]


struct Model
    KZ :: Array{Complex128, 2}
    Rguard :: Array{Float64, 1}
    Tguard :: Array{Float64, 1}
    Kguard :: Array{Float64, 2}
    Wguard :: Array{Float64, 1}
end


function Model(unit::Units.Unit, field::Fields.Field, medium::Media.Medium,
               keys::Dict{String,Any})
    grid = field.grid

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

    return Model(KZ, Rguard, Tguard, Kguard, Wguard)
end


function zstep(dz::Float64, field::Fields.Field, model::Models.Model)
    # Field -> temporal spectrum -----------------------------------------------
    for i=1:field.grid.Nr
        field.S[i, :] = conj(rfft(real(field.E[i, :])))   # time -> frequency
    end

    # Linear propagator --------------------------------------------------------
    for j=1:field.grid.Nw
        field.S[:, j] = Hankel.dht(field.grid.HT, field.S[:, j])
        for i=1:field.grid.Nr
            field.S[i, j] = field.S[i, j] *
                            exp(1im * model.KZ[i, j] * dz) *
                            model.Kguard[i, j]   # angular filter
        end
        field.S[:, j] = Hankel.idht(field.grid.HT, field.S[:, j])
    end

    # Temporal spectrum -> field -----------------------------------------------
    for i=1:field.grid.Nr
        field.S[i, :] = @. field.S[i, :] * model.Wguard   # spectral filter
        Sa = spectrum_real_to_analytic(field.S[i, :], field.grid.Nt)
        field.E[i, :] = fft(Sa) / field.grid.Nt   # frequency -> time
        field.E[i, :] = @. field.E[i, :] * model.Rguard[i] * model.Tguard   # spatial and temporal filters
    end
end


function spectrum_real_to_analytic(S, Nt)
    Sa = zeros(Complex128, Nt)
    if Nt % 2 == 0   # Nt is even
        Sa[1] = S[1]
        Sa[2:div(Nt, 2)] = 2. * S[2:div(Nt, 2)]
        Sa[div(Nt, 2) + 1] = S[div(Nt, 2) + 1]
    else   # Nt is odd
        Sa[1] = S[1]
        Sa[2:div(Nt + 1, 2)] = 2. * S[2:div(Nt + 1, 2)]
    end
    return Sa
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

module PlasmaComponents

using PyCall
@pyimport scipy.constants as sc

import Units
import Grids
import Fields
import Media
import TabularFunctions

const QE = sc.e   # elementary charge [C]
const ME = sc.m_e   # electron mass [kg]
const HBAR = sc.hbar   # the Planck constant (divided by 2*pi) [J*s]


struct Component
    name :: String
    frac :: Float64
    Ui :: Float64
    K :: Float64
    Wava :: Float64
    tf :: TabularFunctions.TabularFunction

    rho0 :: Float64
    rho :: Array{Float64, 1}
    Kdrho :: Array{Float64, 1}
    RI :: Array{Float64, 1}
    keys :: Dict{String, Any}
end


function Component(unit::Units.Unit, grid::Grids.Grid, field::Fields.Field,
                   medium::Media.Medium, rho0::Float64, nuc::Float64,
                   mr::Float64, name::String, frac::Float64, Ui::Float64,
                   fname_tabfunc::String, keys)

    rho0 = rho0 / unit.rho

    Ui = Ui * QE   # eV -> J
    tf = TabularFunctions.TabularFunction(unit, fname_tabfunc)

    K = ceil(Ui / (HBAR * field.w0))   # order of multiphoton ionization

    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    MR = mr * ME   # reduced mass of electron and hole (effective mass)
    sigma = QE^2 / MR * nuc / (nuc^2 + field.w0^2)
    Wava = sigma / Ui * Eu^2 * unit.t

    rho = zeros(Float64, grid.Nt)
    Kdrho = zeros(Float64, grid.Nt)
    RI = zeros(Float64, grid.Nt)

    return Component(name, frac, Ui, K, Wava, tf, rho0, rho, Kdrho, RI, keys)
end


"""
Calculates free charge concentration, its derivative, and ionization rate for a
given medium component.
"""
function free_charge(comp::Component, grid::Grids.Grid,
                     Et::Array{ComplexF64, 1})
    IONARG = comp.keys["IONARG"]
    AVALANCHE = comp.keys["AVALANCHE"]

    comp.rho[1] = 0.
    comp.Kdrho[1] = 0.
    comp.RI[1] = 0.

    @inbounds for i=2:grid.Nt
        if IONARG != 0
            Ival = 0.5 * (abs2(Et[i]) + abs2(Et[i-1]))
        else
            Ival = 0.5 * (real(Et[i])^2 + real(Et[i-1])^2)
        end

        W1 = TabularFunctions.tfvalue(comp.tf, Ival)

        if AVALANCHE != 0
            W2 = comp.Wava * Ival
        else
            W2 = 0.
        end

        if W1 == 0.
            # if no field ionization, then calculate only the avalanche one
            comp.rho[i] = comp.rho[i-1] * exp(W2 * grid.dt)
            comp.Kdrho[i] = 0.
        else
            # without this "if" statement 1/W12 will cause NaN values in Ne
            W12 = W1 - W2
            comp.rho[i] = W1 / W12 * comp.frac * comp.rho0 -
                     (W1 / W12 * comp.frac * comp.rho0 - comp.rho[i-1]) *
                     exp(-W12 * grid.dt)
            comp.Kdrho[i] = comp.K * W1 * (comp.frac * comp.rho0 - comp.rho[i])
        end

        comp.RI[i] = W1
    end
    return nothing
end


end

module Plasmas

import Units
import Grids
import Fields
import Media
import PlasmaComponents
import TabularFunctions


mutable struct Plasma
    rho0 :: Float64
    nuc :: Float64
    mr :: Float64
    components :: Array{PlasmaComponents.Component, 1}
    Ncomp :: Int64
    rho :: Array{Float64, 2}
    Kdrho :: Array{Float64, 2}
    RI :: Array{Float64, 2}
    keys :: Dict{String, Any}
end

function Plasma(unit::Units.Unit, grid::Grids.Grid, field::Fields.Field,
                medium::Media.Medium, rho0::Float64, nuc::Float64, mr::Float64,
                components_dict::Array{Dict{String, Any}, 1},
                keys)
    rho0 = rho0 / unit.rho

    Ncomp = length(components_dict)
    components = Array{PlasmaComponents.Component}(Ncomp)
    for i=1:Ncomp
        comp_dict = components_dict[i]
        name = comp_dict["name"]
        frac = comp_dict["fraction"]
        Ui = comp_dict["ionization_energy"]
        fname_tabfunc = comp_dict["tabular_function"]
        components[i] = PlasmaComponents.Component(unit, field, medium, nuc, mr,
                                                  name, frac, Ui, fname_tabfunc)
    end

    rho = zeros(Float64, (grid.Nr, grid.Nt))
    Kdrho = zeros(Float64, (grid.Nr, grid.Nt))
    RI = zeros(Float64, (grid.Nr, grid.Nt))

    return Plasma(rho0, nuc, mr, components, Ncomp, rho, Kdrho, RI, keys)
end


# function peak_plasma_density(plasma)
#     rhomax = maximum(plasma.rho[:, end])
#     return rhomax
# end
#
#
# function plasma_radius(plasma, grid)
#     rad = 2. * radius(grid.r, plasma.rho[:, end])
#     # Factor 2. because rho(r) is only half of full distribution rho(x)
#     return rad
# end
#
#
# """
# Linear plasma density:
#     lrho = Int[Ne * 2*pi*r*dr],   [De] = 1/m
# """
# function linear_plasma_density(plasma, grid)
#     lrho = 0.
#     for i=1:grid.Nr
#         dr = step(i, grid.r)
#         lrho = lrho + plasma.rho[i, end] * grid.r[i] * dr
#     end
#     lrho = lrho * 2. * pi
#     return lrho
# end


"""
Calculates free charge concentration, its derivative, and ionization rate for
all medium components.
"""
function free_charge(plasma, grid, field)
    plasma.rho .= 0.
    plasma.Kdrho .= 0.
    plasma.RI .= 0.
    for i=1:plasma.Ncomp
        comp = plasma.components[i]
        for j=1:grid.Nr
            Et = field.E[j, :]
            rhot, drhot, RIt = free_charge_component(plasma, grid, Et, comp)
            plasma.rho[j, :] = plasma.rho[j, :] + rhot
            plasma.Kdrho[j, :] = plasma.Kdrho[j, :] + comp.K * drhot
            plasma.RI[j, :] = plasma.RI[j, :] + RIt
        end
    end
    field.rho = plasma.rho[:, end]
end


"""
Calculates free charge concentration, its derivative, and ionization rate for a
specific medium component.
"""
function free_charge_component(plasma, grid, Et, comp)
    IONARG = plasma.keys["IONARG"]
    AVALANCHE = plasma.keys["AVALANCHE"]

    if IONARG != 0
        It = @. abs2(Et)
    else
        It = @. real(Et)^2
    end

    rho = zeros(Float64, grid.Nt)
    drho = zeros(Float64, grid.Nt)
    RI = zeros(Float64, grid.Nt)
    for i=2:grid.Nt
        Ival = 0.5 * (It[i] + It[i-1])
        W1 = TabularFunctions.tfvalue(comp.tf, Ival)
        if AVALANCHE != 0
            W2 = comp.Wava * Ival
        else
            W2 = 0.
        end
        if W1 == 0.
            # if no field ionization, then calculate only the avalanche one
            rho[i] = rho[i-1] * exp(W2 * grid.dt)
            drho[i] = 0.
        else
            # without this "if" statement 1/W12 will cause NaN values in Ne
            W12 = W1 - W2
            rho[i] = W1 / W12 * comp.frac * plasma.rho0 -
                     (W1 / W12 * comp.frac * plasma.rho0 - rho[i-1]) *
                     exp(-W12 * grid.dt)
            drho[i] = W1 * (comp.frac * plasma.rho0 - rho[i])
        end
        RI[i] = W1
    end
    return rho, drho, RI
end


end

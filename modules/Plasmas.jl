module Plasmas

import Units
import Grids
import Fields
import Media
import PlasmaComponents


struct Plasma
    nuc :: Float64
    mr :: Float64
    components :: Array{PlasmaComponents.Component, 1}
    Ncomp :: Int64
    rho :: Array{Float64, 2}
    Kdrho :: Array{Float64, 2}
    RI :: Array{Float64, 2}
end

function Plasma(unit::Units.Unit, grid::Grids.Grid, field::Fields.Field,
                medium::Media.Medium, rho0::Float64, nuc::Float64, mr::Float64,
                components_dict::Array{Dict{String, Any}, 1},
                keys)

    Ncomp = length(components_dict)
    components = Array{PlasmaComponents.Component}(Ncomp)
    for i=1:Ncomp
        comp_dict = components_dict[i]
        name = comp_dict["name"]
        frac = comp_dict["fraction"]
        Ui = comp_dict["ionization_energy"]
        fname_tabfunc = comp_dict["tabular_function"]
        components[i] = PlasmaComponents.Component(unit, grid, field, medium,
                                                   rho0, nuc, mr, name, frac,
                                                   Ui, fname_tabfunc, keys)
    end

    rho = zeros(Float64, (grid.Nr, grid.Nt))
    Kdrho = zeros(Float64, (grid.Nr, grid.Nt))
    RI = zeros(Float64, (grid.Nr, grid.Nt))

    return Plasma(nuc, mr, components, Ncomp, rho, Kdrho, RI)
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
    Et = zeros(Complex128, grid.Nt)

    @inbounds @. plasma.rho = 0.
    @inbounds @. plasma.Kdrho = 0.
    @inbounds @. plasma.RI = 0.
    for comp in plasma.components
        @inbounds for i=1:grid.Nr
            @views @. Et = field.E[i, :]
            PlasmaComponents.free_charge(comp, grid, Et)
            @views @. plasma.rho[i, :] = plasma.rho[i, :] + comp.rho
            @views @. plasma.Kdrho[i, :] = plasma.Kdrho[i, :] + comp.Kdrho
            @views @. plasma.RI[i, :] = plasma.RI[i, :] + comp.RI
        end
    end
    @inbounds @views @. field.rho = plasma.rho[:, end]
end


end

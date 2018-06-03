module MediaComponents

using PyCall
@pyimport scipy.constants as sc

import Units
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
end


function Component(unit::Units.Unit, field::Fields.Field, medium::Media.Medium,
                   nuc::Float64, mr::Float64, name::String, frac::Float64,
                   Ui::Float64, fname_tabfunc::String)

    Ui = Ui * QE   # eV -> J
    tf = TabularFunctions.TabularFunction(unit, fname_tabfunc)

    K = ceil(Ui / (HBAR * field.w0))   # order of multiphoton ionization

    n0 = real(Media.refractive_index(medium, field.w0))
    Eu = Units.E(unit, n0)
    MR = mr * ME   # reduced mass of electron and hole (effective mass)
    sigma = QE^2 / MR * nuc / (nuc^2 + field.w0^2)
    Wava = sigma / Ui * Eu^2 * unit.t

    return Component(name, frac, Ui, K, Wava, tf)
end


end

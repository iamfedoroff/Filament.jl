module Units

import PyCall

scipy_constants = PyCall.pyimport("scipy.constants")
const C0 = scipy_constants.c   # speed of light in vacuum
const EPS0 = scipy_constants.epsilon_0   # the electric constant (vacuum permittivity) [F/m]


abstract type Unit end


struct UnitR <: Unit
    r :: Float64
    k :: Float64
    z :: Float64
    I :: Float64
end


struct UnitRT <: Unit
    r :: Float64
    k :: Float64
    z :: Float64
    t :: Float64
    w :: Float64
    lam :: Float64
    I :: Float64
    rho :: Float64
end


function Unit(ru::Float64, zu::Float64, Iu::Float64)
    ku = 1. / ru
    return UnitR(ru, ku, zu, Iu)
end


function Unit(ru::Float64, zu::Float64, tu::Float64, Iu::Float64, rhou::Float64)
    ku = 1. / ru
    wu = 1. / tu
    lamu = tu
    return UnitRT(ru, ku, zu, tu, wu, lamu, Iu, rhou)
end


"""
Units of electric field (depends on refractive index n)
"""
function E(unit::Unit, n)
    Eu = sqrt(unit.I / (0.5 * real(n) * EPS0 * C0))
    return Eu
end


end

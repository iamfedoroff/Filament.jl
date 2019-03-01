module Units

import PyCall

scipy_constants = PyCall.pyimport("scipy.constants")
const C0 = scipy_constants.c   # speed of light in vacuum
const EPS0 = scipy_constants.epsilon_0   # the electric constant (vacuum permittivity) [F/m]


struct Unit
    r :: Float64
    z :: Float64
    t :: Float64
    I :: Float64
    rho :: Float64

    k :: Float64
    w :: Float64
    lam :: Float64
end


function Unit(ru, zu, tu, Iu, rhou)
    ku = 1. / ru
    wu = 1. / tu
    lamu = tu
    return Unit(ru, zu, tu, Iu, rhou, ku, wu, lamu)
end


"""
Units of electric field (depends on refractive index n)
"""
function E(unit, n)
    Eu = sqrt(unit.I / (0.5 * real(n) * EPS0 * C0))
    return Eu
end


end

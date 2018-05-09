module Units

using PyCall
@pyimport scipy.constants as sc

C0 = sc.c   # speed of light in vacuum
EPS0 = sc.epsilon_0   # the electric constant (vacuum permittivity) [F/m]


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

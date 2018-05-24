module Media

using PyCall
@pyimport scipy.constants as sc

import ForwardDiff

const C0 = sc.c   # speed of light in vacuum
const EPS0 = sc.epsilon_0   # the electric constant (vacuum permittivity) [F/m]


struct Medium
    permittivity :: Function
    permeability :: Function
    n2 :: Float64
    rho0 :: Float64
    nuc :: Float64
    mr :: Float64
end


function refractive_index(medium, w)
    eps = medium.permittivity(w)
    mu = medium.permeability(w)
    n = sqrt(eps * mu)
    return n
end


function beta_func(medium, w)
    n = refractive_index(medium, w)
    beta = n * w / C0
    return beta
end


function k_func(medium, w)
    beta = beta_func(medium, w)
    k = real(beta)
    return k
end


function k1_func(medium, w)
    func(w) = k_func(medium, w)
    k1 = ForwardDiff.derivative(func, w)
    return k1
end


function k2_func(medium, w)
    func(w) = k1_func(medium, w)
    k2 = ForwardDiff.derivative(func, w)
    return k2
end


function k3_func(medium, w)
    func(w) = k2_func(medium, w)
    k3 = ForwardDiff.derivative(func, w)
    return k3
end


function absorption_coefficient(medium, w)
    beta = beta_func(medium, w)
    ga = imag(beta)
    return ga
end


function phase_velocity(medium, w)
    n = refractive_index(medium, w)
    vp = C0 / real(n)
    return vp
end


function group_velocity(medium, w)
    k1 = k1_func(medium, w)
    vg = 1. / k1
    return vg
end


function diffraction_length(medium, w, a0)
    k = k_func(medium, w)
    Ld = k * a0^2
    return Ld
end


function dispersion_length(medium, w, t0)
    k2 = k2_func(medium, w)
    if k2 == 0.
        Ldisp = Inf
    else
        Ldisp = t0^2 / abs(k2)
    end
    return Ldisp
end


function dispersion_length3(medium, w, t0)
    k3 = k3_func(medium, w)
    if k3 == 0.
        Ldisp3 = Inf
    else
        Ldisp3 = t0^3 / abs(k3)
    end
    return Ldisp3
end


function absorption_length(medium, w)
    ga = absorption_coefficient(medium, w)
    if ga == 0.
        La = Inf
    else
        La = 0.5 / ga
    end
    return La
end


function chi3_func(medium, w)
    n = refractive_index(medium, w)
    n2 = medium.n2
    chi3 = 4. / 3. * real(n)^2 * EPS0 * C0 * n2
    return chi3
end


function critical_power(medium, w)
    Rcr = 3.79
    lam = 2. * pi * C0 / w
    n = refractive_index(medium, w)
    n2 = medium.n2
    Pcr = Rcr * lam^2 / (8. * pi * abs(real(n)) * abs(real(n2)))
    return Pcr
end


function nonlinearity_length(medium, w, I0)
    n2 = medium.n2
    if n2 == 0.
        Lnl = Inf
    else
        Lnl = 1. / (abs(real(n2)) * I0 * w / C0)
    end
    return Lnl
end

"""Self-focusing distance by the Marburger formula (P in watts)."""
function selffocusing_length(medium, w, a0, P)
    Ld = diffraction_length(medium, w, a0)
    PPcr = P / critical_power(medium, w)
    if PPcr > 1.
        zf = 0.367 * Ld / sqrt((sqrt(PPcr) - 0.852)^2 - 0.0219)
    else
        zf = Inf
    end
    return zf
end


end

module Media

# import ForwardDiff

import Constants: C0, EPS0


struct Medium{T<:AbstractFloat}
    permittivity :: Function
    permeability :: Function
    n2 :: T
end


struct NonlinearResponse{T<:Union{AbstractFloat,AbstractArray}, F<:Function}
    Rnl :: T
    calculate :: F
    p :: Tuple
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
    # k1 = ForwardDiff.derivative(func, w)
    k1 = derivative(func, w, 1)
    return k1
end


function k2_func(medium, w)
    # func(w) = k1_func(medium, w)
    # k2 = ForwardDiff.derivative(func, w)
    func(w) = k_func(medium, w)
    k2 = derivative(func, w, 2)
    return k2
end


function k3_func(medium, w)
    # func(w) = k2_func(medium, w)
    # k3 = ForwardDiff.derivative(func, w)
    func(w) = k_func(medium, w)
    k3 = derivative(func, w, 3)
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


"""
N-th derivative of a function f at a point x.

The derivative is found using five-point stencil:
    http://en.wikipedia.org/wiki/Five-point_stencil
Additional info:
    http://en.wikipedia.org/wiki/Finite_difference_coefficients
"""
function derivative(f::Function, x::Float64, n::Int64)
    if x == 0.
        h = 0.01
    else
        h = 0.001 * x
    end

    if n == 1
        res = (f(x - 2. * h) - 8. * f(x - h) + 8. * f(x + h) - f(x + 2. * h)) /
              (12. * h)
    elseif n == 2
        res = (- f(x - 2. * h) + 16. * f(x - h) - 30. * f(x) + 16. * f(x + h) -
                 f(x + 2. * h)) / (12. * h^2)
    elseif n == 3
        res = (- f(x - 2. * h) + 2. * f(x - h) - 2. * f(x + h) +
                 f(x + 2. * h)) / (2. * h^3)
    elseif n == 4
        res = (f(x - 2. * h) - 4. * f(x - h) + 6. * f(x) - 4. * f(x + h) +
               f(x + 2. * h)) / (h^4)
    else
        println("ERROR: Wrong derivative order.")
        exit()
    end
    return res
end


end

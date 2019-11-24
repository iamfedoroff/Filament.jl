z = 0.   # [m] the distance at which the initial condition is defined

lam0 = 800e-9   # [m] central wavelength


function initial_condition(r, t, ru, tu, Iu)
    a0 = 1e-2  # [m] initial beam radius
    t0 = 20e-15  # [s] initial pulse duration
    I0 = 1e12 * 1e4   # [W/m^2] initial intensity

    w0 = 2 * pi * C0 / lam0   # central frequency

    Nr = length(r)
    Nt = length(t)
    E = zeros(ComplexF64, (Nr, Nt))
    for j=1:Nt
    for i=1:Nr
        E[i, j] = sqrt(I0 / Iu) *
                  exp(-0.5 * (r[i] * ru)^2 / a0^2) *
                  exp(-0.5 * (t[j] * tu)^2 / t0^2) *
                  cos(w0 * t[j] * tu)
    end
    end
    return E
end

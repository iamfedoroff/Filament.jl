z = 0.   # [m] the distance at which the initial condition is defined

lam0 = 800e-9   # [m] central wavelength


function initial_condition(t, tu, Iu)
    t0 = 20e-15  # [s] initial pulse duration
    I0 = 1e12 * 1e4   # [W/cm^2] initial intensity

    w0 = 2 * pi * C0 / lam0   # central frequency

    E = @. sqrt(I0 / Iu) * exp(-0.5 * (t * tu)^2 / t0^2) * cos(w0 * t * tu)
    return E
end

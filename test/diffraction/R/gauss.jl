z = 0.   # [m] the distance at which the initial condition is defined

lam0 = 800e-9   # [m] central wavelength


function initial_condition(r, ru, Iu)
    a0 = 1e-3  # [m] initial beam radius
    I0 = 1e12 * 1e4   #  [W/cm^2] initial beam intensity

    E = @. sqrt(I0 / Iu) * exp(-0.5 * (r * ru)^2 / a0^2) + 0im

    # Focusing:
    # f = 0.5   # [m] focal distance
    # n0 = 1.   # refractive index
    # w0 = 2. * pi * C0 / lam0   # central frequency
    # k0 = n0 * w0 / C0   # wavenumber
    # E = @. E * exp(1im * k0 * (r * ru)^2 / (2. * f))

    return E
end

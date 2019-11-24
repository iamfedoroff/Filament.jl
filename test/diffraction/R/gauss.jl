z = 0.   # [m] the distance at which the initial condition is defined

lam0 = 800e-9   # [m] central wavelength


function initial_condition(r, ru, Iu)
    a0 = 1e-3  # [m] initial beam radius
    I0 = 1e12 * 1e4   #  [W/cm^2] initial beam intensity

    E = @. sqrt(I0 / Iu) * exp(-0.5 * (r * ru)^2 / a0^2) + 0im
    return E
end

z = 0.   # [m] the distance at which the initial condition is defined

lam0 = 800e-9   # [m] central wavelength


function initial_condition(x, y, xu, yu, Iu)
    a0 = 1e-3  # [m] initial beam radius
    I0 = 1e12 * 1e4   #  [W/cm^2] initial beam intensity

    Nx = length(x)
    Ny = length(y)
    E = zeros(ComplexF64, (Nx, Ny))
    for j=1:Ny
    for i=1:Nx
        E[i,j] = sqrt(I0 / Iu) *
                 exp(-0.5 * ((x[i] * xu)^2 + (y[j] * yu)^2) / a0^2)
    end
    end
    return E
end

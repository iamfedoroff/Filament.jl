z = 0.   # [m] the distance at which the initial condition is defined

lam0 = 800e-9   # [m] central wavelength


function initial_condition(r, t, ru, tu, Iu)
    a0 = 3e-3 / (2. * sqrt(log(2.)))  # [m] initial beam radius
    t0 = 35e-15 / (2. * sqrt(log(2.)))  # [s] initial pulse duration
    W = 1.5e-3   # [J] initial pulse energy

    w0 = 2. * pi * C0 / lam0   # central frequency
    I0 = W / (pi^1.5 * t0 * a0^2)   # initial pulse intensity

    Nr = length(r)
    Nt = length(t)
    E = zeros(Complex128, (Nr, Nt))
    for j=1:Nt
        for i=1:Nr
            E[i, j] = sqrt(I0 / Iu) *
                      exp(-0.5 * (r[i] * ru)^2 / a0^2) *
                      exp(-0.5 * (t[j] * tu)^2 / t0^2) *
                      cos(w0 * t[j] * tu)
        end
    end

    # Focusing:
    f = 2.   # [m] focal distance
    n0 = 1.   # refractive index
    wu = 1. / tu
    w = 2. * pi * npfft.rfftfreq(Nt, t[2] - t[1])
    for i=1:Nr
        Et = real(E[i, :])
        Ew = FFTW.rfft(Et)
        Ew = @. Ew * exp(1im * n0 * (w * wu) / C0 * (r[i] * ru)^2 / (2. * f))
        Et = FFTW.irfft(Ew, Nt)
        E[i, :] = Et
    end

    return E
end

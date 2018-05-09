module Grids

using PyCall
@pyimport numpy.fft as npfft
@pyimport scipy.constants as sc

import Hankel

C0 = sc.c   # speed of light in vacuum


struct Grid
    geometry :: String

    rmax :: Float64
    Nr :: Int64

    tmin :: Float64
    tmax :: Float64
    Nt :: Int64

    HT :: Hankel.HankelTransform
    r :: Array{Float64, 1}
    dr :: Float64
    v :: Array{Float64, 1}
    k :: Array{Float64, 1}
    dk :: Float64
    kc :: Float64

    t :: Array{Float64, 1}
    dt :: Float64
    f :: Array{Float64, 1}
    Nf :: Int64
    df :: Float64
    fc :: Float64
    w :: Array{Float64, 1}
    Nw :: Int64
    dw :: Float64
    wc :: Float64
    lam :: Array{Float64, 1}
    Nlam :: Int64
end


function Grid(rmax, Nr, tmin, tmax, Nt)
    geometry = "RT"

    HT = Hankel.HankelTransform(rmax, Nr)

    r = HT.r   # radial coordinates
    dr = mean(diff(r))   # spatial step

    v = HT.v   # spatial frequency
    k = 2. * pi * v   # spatial angular frequency
    dk = mean(diff(k))   # spatial frequency step
    kc = 2. * pi * 0.5 / dr   # spatial Nyquist frequency

    t = linspace(tmin, tmax, Nt)   # temporal coordinates
    dt = t[2] - t[1]   # temporal step

    f = npfft.rfftfreq(Nt, dt)   # temporal frequency
    Nf = length(f)   # length of temporal frequency array
    df = f[2] - f[1]   # temporal frequency step
    fc = 0.5 / dt   # temporal Nyquist frequency

    w = 2. * pi * f   # temporal angular frequency
    Nw = length(w)   # length of temporal angular frequency array
    dw = w[2] - w[1]   # temporal angular frequency step
    wc = 2. * pi * fc   # temporal angular Nyquist frequency

    lam = zeros(Nf)   # wavelengths
    for i=1:Nf
        if f == 0.
            lam[i] = Inf
        else
            lam[i] = C0 / f[i]
        end
    end
    Nlam = length(lam)
    # dlam = lam[3] - lam[2]   # wavelength step
    # lamc = 0.5 / self.dlam   # Nyquist wavelength (need check!)

    return Grid(geometry, rmax, Nr, tmin, tmax, Nt,
                HT, r, dr, v, k, dk, kc,
                t, dt, f, Nf, df, fc, w, Nw, dw, wc, lam, Nlam)
end


end

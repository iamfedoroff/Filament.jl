module Grids

import CuArrays

import Hankel
import Fourier

import PyCall
scipy_constants = PyCall.pyimport("scipy.constants")
const C0 = scipy_constants.c   # speed of light in vacuum

const FloatGPU = Float32


abstract type Grid end


struct GridR <: Grid
    geometry :: String

    rmax :: Float64
    Nr :: Int

    HT :: Hankel.HankelTransform

    r :: Array{Float64, 1}
    dr :: Array{Float64, 1}
    rdr :: CuArrays.CuArray{FloatGPU, 1}
    dr_mean :: Float64
    v :: Array{Float64, 1}
    k :: Array{Float64, 1}
    dk_mean :: Float64
    kc :: Float64
end


struct GridRT <: Grid
    geometry :: String

    rmax :: Float64
    Nr :: Int

    tmin :: Float64
    tmax :: Float64
    Nt :: Int

    HT :: Hankel.HankelTransform

    r :: Array{Float64, 1}
    dr :: Array{Float64, 1}
    rdr :: CuArrays.CuArray{FloatGPU, 1}
    dr_mean :: Float64
    v :: Array{Float64, 1}
    k :: Array{Float64, 1}
    dk_mean :: Float64
    kc :: Float64

    t :: Array{Float64, 1}
    dt :: Float64
    f :: Array{Float64, 1}
    Nf :: Int
    df :: Float64
    fc :: Float64
    w :: Array{Float64, 1}
    Nw :: Int
    dw :: Float64
    wc :: Float64
    lam :: Array{Float64, 1}
    Nlam :: Int

    FT :: Fourier.FourierTransform
end


struct GridXY <: Grid
    geometry :: String

    xmin :: Float64
    xmax :: Float64
    Nx :: Int

    ymin :: Float64
    ymax :: Float64
    Ny :: Int

    x :: Array{Float64, 1}
    dx :: Float64
    kx :: Array{Float64, 1}
    Nkx :: Int
    dkx :: Float64
    kxc :: Float64

    y :: Array{Float64, 1}
    dy :: Float64
    ky :: Array{Float64, 1}
    Nky :: Int
    dky :: Float64
    kyc :: Float64

    FT :: Fourier.FourierTransform
end


function Grid(rmax, Nr)
    geometry = "R"

    HT = Hankel.HankelTransform(rmax, Nr)   # Hankel transform

    r = HT.r   # radial coordinates

    # steps for radial coordinate:
    dr = zeros(Nr)
    for i=1:Nr
        dr[i] = step(i, r)
    end

    rdr = r .* dr   # for calculation of spatial integrals
    rdr = CuArrays.CuArray(convert(Array{FloatGPU, 1}, rdr))

    dr_mean = sum(diff(r)) / length(diff(r))   # spatial step

    v = HT.v   # spatial frequency
    k = 2. * pi * v   # spatial angular frequency
    dk_mean = sum(diff(k)) / length(diff(k))   # spatial frequency step
    kc = 2. * pi * 0.5 / dr_mean   # spatial Nyquist frequency

    return GridR(geometry, rmax, Nr,
                 HT, r, dr, rdr, dr_mean, v, k, dk_mean, kc)
end


function Grid(rmax, Nr, tmin, tmax, Nt)
    geometry = "RT"

    HT = Hankel.HankelTransform(rmax, Nr, Nt)   # Hankel transform

    r = HT.r   # radial coordinates

    # steps for radial coordinate:
    dr = zeros(Nr)
    for i=1:Nr
        dr[i] = step(i, r)
    end

    rdr = r .* dr   # for calculation of spatial integrals
    rdr = CuArrays.CuArray(convert(Array{FloatGPU, 1}, rdr))

    dr_mean = sum(diff(r)) / length(diff(r))   # spatial step

    v = HT.v   # spatial frequency
    k = 2. * pi * v   # spatial angular frequency
    dk_mean = sum(diff(k)) / length(diff(k))   # spatial frequency step
    kc = 2. * pi * 0.5 / dr_mean   # spatial Nyquist frequency

    t = range(tmin, tmax, length=Nt)   # temporal coordinates
    dt = t[2] - t[1]   # temporal step

    f = Fourier.rfftfreq(Nt, dt)   # temporal frequency
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

    FT = Fourier.FourierTransformRT(Nr, Nt)   # Fourier transform

    return GridRT(geometry, rmax, Nr, tmin, tmax, Nt,
                  HT, r, dr, rdr, dr_mean, v, k, dk_mean, kc,
                  t, dt, f, Nf, df, fc, w, Nw, dw, wc, lam, Nlam, FT)
end


function Grid(xmin, xmax, Nx, ymin, ymax, Ny)
    geometry = "XY"

    x = range(xmin, xmax, length=Nx)   # x spatial coordinates
    dx = x[2] - x[1]   # x spatial step

    y = range(ymin, ymax, length=Ny)   # y spatial coordinates
    dy = y[2] - y[1]   # y spatial step

    kx = 2. * pi * Fourier.fftfreq(Nx, dx)   # x angular spatial frequency
    Nkx = length(kx)
    dkx = kx[2] - kx[1]   # x angular spatial frequency step
    kxc = 2. * pi * 0.5 / dkx   # x angular spatial Nyquist frequency

    ky = 2. * pi * Fourier.fftfreq(Ny, dy)   # y angular spatial frequency
    Nky = length(ky)
    dky = ky[2] - ky[1]   # y angular spatial frequency step
    kyc = 2. * pi * 0.5 / dky   # y angular spatial Nyquist frequency

    FT = Fourier.FourierTransformXY(Nx, Ny)   # Fourier transform

    return GridXY(geometry, xmin, xmax, Nx, ymin, ymax, Ny,
                  x, dx, kx, Nkx, dkx, kxc, y, dy, ky, Nky, dky, kyc, FT)
end


"""Calculates step dx for a specific index i on a nonuniform grid x."""
function step(i::Int, x::Array{Float64, 1})
    Nx = length(x)
    if i == 1
        dx = x[2] - x[1]
    elseif i == Nx
        dx = x[Nx] - x[Nx - 1]
    else
        dx = 0.5 * (x[i+1] - x[i-1])
    end
    return dx
end


function radius(x::Array{Float64, 1}, y::Array{FloatGPU, 1},
                level::Float64=exp(-1.))
    Nx = length(x)
    ylevel = maximum(y) * level

    radl = 0.
    for i=1:Nx
        if y[i] >= ylevel
            radl = x[i]
            break
        end
    end

    radr = 0.
    for i=Nx:-1:1
        if y[i] >= ylevel
            radr = x[i]
            break
        end
    end

    return 0.5 * (abs(radl) + abs(radr))
end


end
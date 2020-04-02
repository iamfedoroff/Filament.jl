module Grids

import CuArrays

import Constants: FloatGPU
import Fourier
import Hankel
import HankelTransforms


abstract type Grid end


struct GridR{
    I <: Int,
    T <: AbstractFloat,
    U <: AbstractArray{T},
    UG <: AbstractArray{T},
    PH <: HankelTransforms.Plan,
} <: Grid
    rmax :: T
    Nr :: I
    r :: U
    dr :: U
    rdr :: UG
    k :: U
    HT :: PH
end


struct GridT{T<:AbstractFloat, I<:Int} <: Grid
    geometry :: String

    tmin :: T
    tmax :: T
    Nt :: I
    t :: AbstractArray{T, 1}
    dt :: T
    w :: AbstractArray{T, 1}
    Nw :: I

    FT :: Fourier.FourierTransform
end


struct GridRT{T<:AbstractFloat, I<:Int} <: Grid
    geometry :: String

    rmax :: T
    Nr :: I
    r :: AbstractArray{T, 1}
    dr :: AbstractArray{T, 1}
    rdr :: CuArrays.CuArray{FloatGPU, 1}
    k :: AbstractArray{T, 1}

    tmin :: T
    tmax :: T
    Nt :: I
    t :: AbstractArray{T, 1}
    dt :: T
    w :: AbstractArray{T, 1}
    Nw :: I

    HT :: Hankel.HankelTransform
    FT :: Fourier.FourierTransform
end


struct GridXY{T<:AbstractFloat, I<:Int} <: Grid
    geometry :: String

    xmin :: T
    xmax :: T
    Nx :: I
    x :: AbstractArray{T, 1}
    dx :: T
    kx :: AbstractArray{T, 1}

    ymin :: T
    ymax :: T
    Ny :: I
    y :: AbstractArray{T, 1}
    dy :: T
    ky :: AbstractArray{T, 1}

    FT :: Fourier.FourierTransform
end


function Grid(geometry::String, p::Tuple)
    if geometry == "R"
        unit = GridR(p...)
    elseif geometry == "T"
        unit = GridT(p...)
    elseif geometry == "RT"
        unit = GridRT(p...)
    elseif geometry == "XY"
        unit = GridXY(p...)
    elseif geometry == "XYT"
        error("XYT geometry is not implemented yet.")
    else
        error("Wrong grid geometry.")
    end
    return unit
end


function GridR(rmax::T, Nr::Int) where T<:AbstractFloat
    r, dr, rdr, k = _grid_spatial_axial(rmax, Nr)

    E = CuArrays.zeros(Complex{T}, Nr)
    HT = HankelTransforms.plan(rmax, E)

    rdr = CuArrays.CuArray(convert(Array{T, 1}, rdr))

    return GridR(rmax, Nr, r, dr, rdr, k, HT)
end


function GridT(tmin, tmax, Nt)
    geometry = "T"
    t, dt, w, Nw = _grid_temporal(tmin, tmax, Nt)
    FT = Fourier.FourierTransformT(Nt)   # Fourier transform
    return GridT(geometry, tmin, tmax, Nt, t, dt, w, Nw, FT)
end


function GridRT(rmax, Nr, tmin, tmax, Nt)
    geometry = "RT"
    HT = Hankel.HankelTransform(rmax, Nr, Nt)   # Hankel transform
    r, dr, rdr, k = _grid_spatial_axial(HT)
    t, dt, w, Nw = _grid_temporal(tmin, tmax, Nt)
    FT = Fourier.FourierTransformRT(Nr, Nt)   # Fourier transform

    rdr = CuArrays.CuArray(convert(Array{FloatGPU, 1}, rdr))

    return GridRT(
        geometry, rmax, Nr, r, dr, rdr, k, tmin, tmax, Nt, t, dt, w, Nw, HT, FT,
    )
end


function GridXY(xmin, xmax, Nx, ymin, ymax, Ny)
    geometry = "XY"
    x, dx, kx = _grid_spatial_rectangular(xmin, xmax, Nx)
    y, dy, ky = _grid_spatial_rectangular(ymin, ymax, Ny)
    FT = Fourier.FourierTransformXY(Nx, Ny)   # Fourier transform
    return GridXY(
        geometry, xmin, xmax, Nx, x, dx, kx, ymin, ymax, Ny, y, dy, ky, FT,
    )
end


function _grid_spatial_rectangular(
    xmin::T, xmax::T, Nx::Int,
) where T<:AbstractFloat
    x = range(xmin, xmax, length=Nx)   # grid coordinates
    dx = x[2] - x[1]   # step
    kx = 2 * pi * Fourier.fftfreq(Nx, dx)   # angular frequency
    return x, dx, kx
end


function _grid_spatial_axial(HT::Hankel.HankelTransform)
    r = HT.r   # grid coordinates
    k = 2 * pi * HT.v   # angular frequency

    # nonuniform steps:
    Nr = length(r)
    dr = zeros(Nr)
    for i=1:Nr
        dr[i] = _step(i, r)
    end
    rdr = @. r * dr   # for calculation of spatial integrals

    return r, dr, rdr, k
end
function _grid_spatial_axial(rmax::T, Nr::Int) where T<:AbstractFloat
    r = HankelTransforms.htcoord(rmax, Nr)
    v = HankelTransforms.htfreq(rmax, Nr)
    k = convert(T, 2 * pi) * v   # angular frequency

    # nonuniform steps:
    Nr = length(r)
    dr = zeros(T, Nr)
    for i=1:Nr
        dr[i] = _step(i, r)
    end
    rdr = @. r * dr   # for calculation of spatial integrals

    return r, dr, rdr, k
end


function _grid_temporal(tmin::T, tmax::T, Nt::Int) where T<:AbstractFloat
    t = range(tmin, tmax, length=Nt)   # grid coordinates
    dt = t[2] - t[1]   # step
    w = 2 * pi * Fourier.rfftfreq(Nt, dt)   # angular frequency
    Nw = length(w)   # length of angular frequency array
    return t, dt, w, Nw
end


"""Calculates step dx for a specific index i on a nonuniform grid x."""
function _step(i::Int, x::AbstractArray{T, 1}) where T
    Nx = length(x)
    if i == 1
        dx = x[2] - x[1]
    elseif i == Nx
        dx = x[Nx] - x[Nx - 1]
    else
        dx = (x[i+1] - x[i-1]) / 2
    end
    return dx
end


function radius(
    x::AbstractArray, y::AbstractArray, level::AbstractFloat=exp(-1),
)
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

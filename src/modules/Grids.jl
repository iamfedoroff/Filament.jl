module Grids

import FFTW
import HankelTransforms


abstract type Grid end


function Grid(geometry::String, p::Tuple)
    if geometry == "R"
        grid = GridR(p...)
    elseif geometry == "Rn"
        grid = GridRn(p...)
    elseif geometry == "T"
        grid = GridT(p...)
    elseif geometry == "RT"
        grid = GridRT(p...)
    elseif geometry == "XY"
        grid = GridXY(p...)
    elseif geometry == "XYT"
        grid = GridXYT(p...)
    else
        error("Wrong grid geometry.")
    end
    return grid
end


# ******************************************************************************
# R
# ******************************************************************************
struct GridR{T} <: Grid
    rmax :: T
    Nr :: Int
    r :: Vector{T}
    dr :: Vector{T}
    k :: Vector{T}
end


function GridR(rmax::T, Nr::Int) where T<:AbstractFloat
    r, dr, k = _grid_spatial_axial(rmax, Nr)
    return GridR{T}(rmax, Nr, r, dr, k)
end


# ------------------------------------------------------------------------------
struct GridRn{T} <: Grid
    rmax :: T
    Nr :: Int
    r :: Vector{T}
    dr :: Vector{T}
end


function GridRn(rmax::T, Nr::Int, scomp::AbstractFloat) where T<:AbstractFloat
    NR = Int(round(scomp * Nr))
    p = 4
    h = rmax / Nr
    a = (rmax - h * (NR - 1)) / (NR - 1)^p
    ii = range(1, NR, length=NR)
    R = @. h * (ii - 1) + a * (ii - 1)^p
    R = @. convert(T, R)

    dR = zeros(T, NR)
    for i=1:NR
        dR[i] = _step(i, R)
    end

    return GridRn{T}(rmax, NR, R, dR)
end


# ******************************************************************************
# T
# ******************************************************************************
struct GridT{T} <: Grid
    tmin :: T
    tmax :: T
    Nt :: Int
    t :: StepRangeLen{T}
    dt :: T
    w :: Vector{T}
end


function GridT(tmin::T, tmax::T, Nt::Int) where T<:AbstractFloat
    t, dt, w = _grid_temporal(tmin, tmax, Nt)
    return GridT{T}(tmin, tmax, Nt, t, dt, w)
end


# ******************************************************************************
# RT
# ******************************************************************************
struct GridRT{T} <: Grid
    rmax :: T
    Nr :: Int
    r :: Vector{T}
    dr :: Vector{T}
    k :: Vector{T}

    tmin :: T
    tmax :: T
    Nt :: Int
    t :: StepRangeLen{T}
    dt :: T
    w :: Vector{T}
end


function GridRT(
    rmax::T, Nr::I, tmin::T, tmax::T, Nt::I,
) where {I<:Int, T<:AbstractFloat}
    r, dr, k = _grid_spatial_axial(rmax, Nr)
    t, dt, w = _grid_temporal(tmin, tmax, Nt)
    return GridRT{T}(
        rmax, Nr, r, dr, k, tmin, tmax, Nt, t, dt, w,
    )
end


# ******************************************************************************
# XY
# ******************************************************************************
struct GridXY{T} <: Grid
    xmin :: T
    xmax :: T
    Nx :: Int
    x :: StepRangeLen{T}
    dx :: T
    kx :: Vector{T}

    ymin :: T
    ymax :: T
    Ny :: Int
    y :: StepRangeLen{T}
    dy :: T
    ky :: Vector{T}
end


function GridXY(
    xmin::T, xmax::T, Nx::I, ymin::T, ymax::T, Ny::I,
) where {I<:Int, T<:AbstractFloat}
    x, dx, kx = _grid_spatial_rectangular(xmin, xmax, Nx)
    y, dy, ky = _grid_spatial_rectangular(ymin, ymax, Ny)
    return GridXY{T}(
        xmin, xmax, Nx, x, dx, kx, ymin, ymax, Ny, y, dy, ky,
    )
end


# ******************************************************************************
# XYT
# ******************************************************************************
struct GridXYT{T} <: Grid
    xmin :: T
    xmax :: T
    Nx :: Int
    x :: StepRangeLen{T}
    dx :: T
    kx :: Vector{T}

    ymin :: T
    ymax :: T
    Ny :: Int
    y :: StepRangeLen{T}
    dy :: T
    ky :: Vector{T}

    tmin :: T
    tmax :: T
    Nt :: Int
    t :: StepRangeLen{T}
    dt :: T
    w :: Vector{T}
end


function GridXYT(
    xmin::T, xmax::T, Nx::I, ymin::T, ymax::T, Ny::I, tmin::T, tmax::T, Nt::I,
) where {I<:Int, T<:AbstractFloat}
    x, dx, kx = _grid_spatial_rectangular(xmin, xmax, Nx)
    y, dy, ky = _grid_spatial_rectangular(ymin, ymax, Ny)
    t, dt, w = _grid_temporal(tmin, tmax, Nt)
    return GridXYT{T}(
        xmin, xmax, Nx, x, dx, kx, ymin, ymax, Ny, y, dy, ky,
        tmin, tmax, Nt, t, dt, w,
    )
end


# ******************************************************************************
function _grid_spatial_rectangular(xmin, xmax, Nx)
    x = range(xmin, xmax, length=Nx)   # grid coordinates
    dx = x[2] - x[1]   # step
    kx = 2 * pi * FFTW.fftfreq(Nx, 1 / dx)   # angular frequency
    return x, dx, kx
end


function _grid_spatial_axial(rmax, Nr)
    r = HankelTransforms.dhtcoord(rmax, Nr)
    k = 2 * pi * HankelTransforms.dhtfreq(rmax, Nr)   # angular frequency

    # nonuniform steps:
    Nr = length(r)
    dr = zeros(Nr)
    for i=1:Nr
        dr[i] = _step(i, r)
    end
    return r, dr, k
end


function _grid_temporal(tmin, tmax, Nt)
    t = range(tmin, tmax, length=Nt)   # grid coordinates
    dt = t[2] - t[1]   # step
    w = 2 * pi * FFTW.fftfreq(Nt, 1 / dt)   # angular frequency
    return t, dt, w
end


"""Calculates step dx for a specific index i on a nonuniform grid x."""
function _step(i::Int, x::Vector)
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


end

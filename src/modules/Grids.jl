module Grids

import FFTW
import HankelTransforms


abstract type Grid end


function Grid(geometry::String, p::Tuple)
    if geometry == "R"
        grid = GridR(p...)
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
struct GridR{
    I<:Int,
    T<:AbstractFloat,
    A<:AbstractArray{T},
} <: Grid
    rmax :: T
    Nr :: I
    r :: A
    dr :: A
    k :: A
end


function GridR(rmax::T, Nr::Int) where T<:AbstractFloat
    r, dr, k = _grid_spatial_axial(rmax, Nr)
    return GridR(rmax, Nr, r, dr, k)
end


# ******************************************************************************
# T
# ******************************************************************************
struct GridT{
    I<:Int,
    T<:AbstractFloat,
    A<:AbstractArray{T},
    AR<:AbstractArray{T},
} <: Grid
    tmin :: T
    tmax :: T
    Nt :: I
    t :: AR
    dt :: T
    w :: A
end


function GridT(tmin::T, tmax::T, Nt::Int) where T<:AbstractFloat
    t, dt, w = _grid_temporal(tmin, tmax, Nt)
    return GridT(tmin, tmax, Nt, t, dt, w)
end


# ******************************************************************************
# RT
# ******************************************************************************
struct GridRT{
    I<:Int,
    T<:AbstractFloat,
    A<:AbstractArray{T},
    AR<:AbstractArray{T},
} <: Grid
    rmax :: T
    Nr :: I
    r :: A
    dr :: A
    k :: A

    tmin :: T
    tmax :: T
    Nt :: I
    t :: AR
    dt :: T
    w :: A
end


function GridRT(
    rmax::T, Nr::I, tmin::T, tmax::T, Nt::I,
) where {I<:Int, T<:AbstractFloat}
    r, dr, k = _grid_spatial_axial(rmax, Nr)
    t, dt, w = _grid_temporal(tmin, tmax, Nt)
    return GridRT(
        rmax, Nr, r, dr, k, tmin, tmax, Nt, t, dt, w,
    )
end


# ******************************************************************************
# XY
# ******************************************************************************
struct GridXY{
    I<:Int,
    T<:AbstractFloat,
    A<:AbstractArray{T},
    AR<:AbstractArray{T},
} <: Grid
    xmin :: T
    xmax :: T
    Nx :: I
    x :: AR
    dx :: T
    kx :: A

    ymin :: T
    ymax :: T
    Ny :: I
    y :: AR
    dy :: T
    ky :: A
end


function GridXY(
    xmin::T, xmax::T, Nx::I, ymin::T, ymax::T, Ny::I,
) where {I<:Int, T<:AbstractFloat}
    x, dx, kx = _grid_spatial_rectangular(xmin, xmax, Nx)
    y, dy, ky = _grid_spatial_rectangular(ymin, ymax, Ny)
    return GridXY(
        xmin, xmax, Nx, x, dx, kx, ymin, ymax, Ny, y, dy, ky,
    )
end


# ******************************************************************************
# XYT
# ******************************************************************************
struct GridXYT{
    I<:Int,
    T<:AbstractFloat,
    A<:AbstractArray{T},
    AR<:AbstractArray{T},
} <: Grid
    xmin :: T
    xmax :: T
    Nx :: I
    x :: AR
    dx :: T
    kx :: A

    ymin :: T
    ymax :: T
    Ny :: I
    y :: AR
    dy :: T
    ky :: A

    tmin :: T
    tmax :: T
    Nt :: I
    t :: AR
    dt :: T
    w :: A
end


function GridXYT(
    xmin::T, xmax::T, Nx::I, ymin::T, ymax::T, Ny::I, tmin::T, tmax::T, Nt::I,
) where {I<:Int, T<:AbstractFloat}
    x, dx, kx = _grid_spatial_rectangular(xmin, xmax, Nx)
    y, dy, ky = _grid_spatial_rectangular(ymin, ymax, Ny)
    t, dt, w = _grid_temporal(tmin, tmax, Nt)
    return GridXYT(
        xmin, xmax, Nx, x, dx, kx, ymin, ymax, Ny, y, dy, ky,
        tmin, tmax, Nt, t, dt, w,
    )
end


# ******************************************************************************
function _grid_spatial_rectangular(
    xmin::T, xmax::T, Nx::Int,
) where T<:AbstractFloat
    x = range(xmin, xmax, length=Nx)   # grid coordinates
    dx = x[2] - x[1]   # step
    kx = convert(T, 2 * pi) * FFTW.fftfreq(Nx, 1 / dx)   # angular frequency
    return x, dx, kx
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
    return r, dr, k
end


function _grid_temporal(tmin::T, tmax::T, Nt::Int) where T<:AbstractFloat
    t = range(tmin, tmax, length=Nt)   # grid coordinates
    dt = t[2] - t[1]   # step
    w = convert(T, 2 * pi) * FFTW.fftfreq(Nt, 1 / dt)   # angular frequency
    return t, dt, w
end


"""Calculates step dx for a specific index i on a nonuniform grid x."""
function _step(i::Int, x::AbstractArray)
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

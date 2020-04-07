module Grids

import HankelTransforms

import Fourier


abstract type Grid end


struct GridR{
    I<:Int,
    T<:AbstractFloat,
    U<:AbstractArray{T},
} <: Grid
    rmax :: T
    Nr :: I
    r :: U
    dr :: U
    k :: U
end


struct GridT{
    I<:Int,
    T<:AbstractFloat,
    U<:AbstractArray{T},
    UT<:AbstractArray{T},
} <: Grid
    tmin :: T
    tmax :: T
    Nt :: I
    t :: UT
    dt :: T
    w :: U
    Nw :: I
end


struct GridRT{
    I<:Int,
    T<:AbstractFloat,
    U<:AbstractArray{T},
    UT<:AbstractArray{T},
} <: Grid
    rmax :: T
    Nr :: I
    r :: U
    dr :: U
    k :: U

    tmin :: T
    tmax :: T
    Nt :: I
    t :: UT
    dt :: T
    w :: U
    Nw :: I
end


struct GridXY{
    I<:Int,
    T<:AbstractFloat,
    U<:AbstractArray{T},
    UK<:AbstractArray{T},
} <: Grid
    xmin :: T
    xmax :: T
    Nx :: I
    x :: U
    dx :: T
    kx :: UK

    ymin :: T
    ymax :: T
    Ny :: I
    y :: U
    dy :: T
    ky :: UK
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
    r, dr, k = _grid_spatial_axial(rmax, Nr)
    return GridR(rmax, Nr, r, dr, k)
end


function GridT(tmin::T, tmax::T, Nt::Int) where T<:AbstractFloat
    t, dt, w, Nw = _grid_temporal(tmin, tmax, Nt)
    return GridT(tmin, tmax, Nt, t, dt, w, Nw)
end


function GridRT(
    rmax::T, Nr::I, tmin::T, tmax::T, Nt::I,
) where {I<:Int, T<:AbstractFloat}
    r, dr, k = _grid_spatial_axial(rmax, Nr)
    t, dt, w, Nw = _grid_temporal(tmin, tmax, Nt)
    return GridRT(
        rmax, Nr, r, dr, k, tmin, tmax, Nt, t, dt, w, Nw,
    )
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


function _grid_spatial_rectangular(
    xmin::T, xmax::T, Nx::Int,
) where T<:AbstractFloat
    x = range(xmin, xmax, length=Nx)   # grid coordinates
    dx = x[2] - x[1]   # step
    kx = convert(T, 2 * pi) * Fourier.fftfreq(Nx, dx)   # angular frequency
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
    w = convert(T, 2 * pi) * Fourier.rfftfreq(Nt, dt)   # angular frequency
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


end

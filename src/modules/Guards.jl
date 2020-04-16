module Guards

import CuArrays
import CUDAnative

import Constants: FloatGPU, MAX_THREADS_PER_BLOCK
import Fields
import Grids
import Media
import Units


abstract type Guard end


# ******************************************************************************
# R
# ******************************************************************************
struct GuardR{A<:AbstractArray} <: Guard
    R :: A
    K :: A
end


function Guard(
    unit::Units.UnitR,
    grid::Grids.GridR,
    field::Fields.FieldR,
    medium::Media.Medium,
    rguard::T,
    kguard::T,
) where T<:AbstractFloat
    # Spatial guard filter:
    Rguard = guard_window(grid.r, rguard, mode="right")
    Rguard = CuArrays.CuArray{T}(Rguard)

    # Angular guard filter:
    k0 = Media.k_func(medium, field.w0)
    kmax = k0 * sind(kguard)
    Kguard = @. exp(-((grid.k * unit.k)^2 / kmax^2)^20)
    Kguard = CuArrays.CuArray{T}(Kguard)

    return GuardR(Rguard, Kguard)
end


function apply_field_filter!(E::AbstractArray{T, 1}, guard::GuardR) where T
    @. E = E * guard.R
    return nothing
end


function apply_spectral_filter!(E::AbstractArray{T, 1}, guard::GuardR) where T
    @. E = E * guard.K
    return nothing
end


# ******************************************************************************
# T
# ******************************************************************************
struct GuardT{A<:AbstractArray} <: Guard
    T :: A
    W :: A
end


function Guard(
    unit::Units.UnitT,
    grid::Grids.GridT,
    field::Fields.FieldT,
    medium::Media.Medium,
    tguard::T,
    wguard::T,
) where T<:AbstractFloat
    # Temporal guard filter:
    Tguard = guard_window(grid.t, tguard, mode="both")

    # Frequency guard filter:
    Wguard = @. exp(-((grid.w * unit.w)^2 / wguard^2)^20)

    return GuardT(Tguard, Wguard)
end


function apply_field_filter!(E::AbstractArray{T, 1}, guard::GuardT) where T
    @. E = E * guard.T
    return nothing
end


function apply_spectral_filter!(E::AbstractArray{T, 1}, guard::GuardT) where T
    @. E = E * guard.W
    return nothing
end


# ******************************************************************************
# RT
# ******************************************************************************
struct GuardRT{A<:AbstractArray} <: Guard
    R :: A
    K :: A
    T :: A
    W :: A
end


function Guard(
    unit::Units.UnitRT,
    grid::Grids.GridRT,
    field::Fields.FieldRT,
    medium::Media.Medium,
    rguard::T,
    tguard::T,
    kguard::T,
    wguard::T,
) where T<:AbstractFloat
    # Spatial guard filter:
    Rguard = guard_window(grid.r, rguard, mode="right")
    Rguard = CuArrays.CuArray{T}(Rguard)

    # Temporal guard filter:
    Tguard = guard_window(grid.t, tguard, mode="both")
    Tguard = CuArrays.CuArray{T}(Tguard)

    # Frequency guard filter:
    Wguard = @. exp(-((grid.w * unit.w)^2 / wguard^2)^20)
    Wguard = CuArrays.CuArray{T}(Wguard)

    # Angular guard filter:
    k0 = Media.k_func(medium, field.w0)
    kmax = k0 * sind(kguard)
    Kguard = @. exp(-((grid.k * unit.k)^2 / kmax^2)^20)
    Kguard = CuArrays.CuArray{T}(Kguard)

    return GuardRT(Rguard, Kguard, Tguard, Wguard)
end


function apply_field_filter!(E::AbstractArray{T, 2}, guard::GuardRT) where T
    N = length(E)
    nth = min(N, MAX_THREADS_PER_BLOCK)
    nbl = cld(N, nth)
    @CUDAnative.cuda blocks=nbl threads=nth kernel!(E, guard.R, guard.T)
end


function apply_spectral_filter!(E::AbstractArray{T, 2}, guard::GuardRT) where T
    N = length(E)
    nth = min(N, MAX_THREADS_PER_BLOCK)
    nbl = cld(N, nth)
    @CUDAnative.cuda blocks=nbl threads=nth kernel!(E, guard.K, guard.W)
end


# ******************************************************************************
# XY
# ******************************************************************************
struct GuardXY{A<:AbstractArray} <: Guard
    X :: A
    Y :: A
    KX :: A
    KY :: A
end


function Guard(
    unit::Units.UnitXY,
    grid::Grids.GridXY,
    field::Fields.Field,
    medium::Media.Medium,
    xguard::T,
    yguard::T,
    kxguard::T,
    kyguard::T,
) where T<:AbstractFloat
    # Spatial guard filters:
    Xguard = guard_window(grid.x, xguard, mode="both")
    Xguard = CuArrays.CuArray{FloatGPU}(Xguard)

    Yguard = guard_window(grid.y, yguard, mode="both")
    Yguard = CuArrays.CuArray{FloatGPU}(Yguard)

    # Angular guard filters:
    k0 = Media.k_func(medium, field.w0)
    kxmax = k0 * sind(kxguard)
    kymax = k0 * sind(kyguard)

    KXguard = @. exp(-((grid.kx * unit.kx)^2 / kxmax^2)^20)
    KXguard = CuArrays.CuArray{FloatGPU}(KXguard)

    KYguard = @. exp(-((grid.ky * unit.ky)^2 / kymax^2)^20)
    KYguard = CuArrays.CuArray{FloatGPU}(KYguard)

    return GuardXY(Xguard, Yguard, KXguard, KYguard)
end


function apply_field_filter!(E::AbstractArray{T, 2}, guard::GuardXY) where T
    N = length(E)
    nth = min(N, MAX_THREADS_PER_BLOCK)
    nbl = cld(N, nth)
    @CUDAnative.cuda blocks=nbl threads=nth kernel!(E, guard.X, guard.Y)
    return nothing
end


function apply_spectral_filter!(E::AbstractArray{T, 2}, guard::GuardXY) where T
    N = length(E)
    nth = min(N, MAX_THREADS_PER_BLOCK)
    nbl = cld(N, nth)
    @CUDAnative.cuda blocks=nbl threads=nth kernel!(E, guard.KX, guard.KY)
    return nothing
end


# ******************************************************************************
"""
Lossy guard window at the ends of grid coordinate.

    x: grid coordinate
    guard_width: the width of the lossy guard window
    mode: "left" - lossy guard only on the left end of the grid
          "right" - lossy guard only on the right end of the grid
          "both" - lossy guard on both ends of the grid
"""
function guard_window(
    x::AbstractArray{T, 1}, guard_width::T; mode="both",
) where T
    @assert guard_width >= 0
    @assert mode in ["left", "right", "both"]

    if mode in ["left", "right"]
        @assert guard_width <= x[end] - x[1]
    else
        @assert guard_width <= 0.5 * (x[end] - x[1])
    end

    Nx = length(x)

    if guard_width == 0
        guard = ones(Nx)
    else
        width = 0.5 * guard_width

        # Left guard
        guard_xmin = x[1]
        guard_xmax = x[2] + guard_width
        gauss1 = zeros(Nx)
        gauss2 = ones(Nx)
        for i=1:Nx
            if x[i] >= guard_xmin
                gauss1[i] = 1 - exp(-((x[i] - guard_xmin) / width)^6)
            end
            if x[i] <= guard_xmax
                gauss2[i] = exp(-((x[i] - guard_xmax) / width)^6)
            end
        end
        guard_left = 0.5 * (gauss1 + gauss2)

        # Right guard
        guard_xmin = x[end] - guard_width
        guard_xmax = x[end]
        gauss1 = ones(Nx)
        gauss2 = zeros(Nx)
        for i=1:Nx
            if x[i] >= guard_xmin
                gauss1[i] = exp(-((x[i] - guard_xmin) / width)^6)
            end
            if x[i] <= guard_xmax
                gauss2[i] = 1 - exp(-((x[i] - guard_xmax) / width)^6)
            end
        end
        guard_right = 0.5 * (gauss1 + gauss2)

        # Result guard:
        if mode == "left"
            guard = guard_left
        elseif mode == "right"
            guard = guard_right
        elseif mode == "both"
            guard = @. guard_left + guard_right - 1
        end
    end
    return guard
end


function kernel!(F, A, B)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x +
         CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    N1, N2 = size(F)
    cartesian = CartesianIndices((N1, N2))
    for k=id:stride:N1*N2
        i = cartesian[k][1]
        j = cartesian[k][2]
        F[i, j] = F[i, j] * A[i] * B[j]
    end
    return nothing
end


end

module Guards

import CUDAnative
import CuArrays
import CUDAdrv

import Units
import Grids
import Media

const FloatGPU = Float32
const ComplexGPU = ComplexF32


abstract type Guard end


struct GuardR{T} <: Guard
    R :: CuArrays.CuArray{T, 1}
    K :: CuArrays.CuArray{T, 1}
end


struct GuardRT{T} <: Guard
    R :: CuArrays.CuArray{T, 1}
    K :: CuArrays.CuArray{T, 2}
    T :: CuArrays.CuArray{T, 1}
    W :: CuArrays.CuArray{T, 1}
    nthreadsNt :: Int
    nthreadsNrNt :: Int
    nthreadsNrNw :: Int
    nblocksNt :: Int
    nblocksNrNt :: Int
    nblocksNrNw :: Int
end


struct GuardXY{T} <: Guard
    X :: CuArrays.CuArray{T, 1}
    Y :: CuArrays.CuArray{T, 1}
    KX :: CuArrays.CuArray{T, 1}
    KY :: CuArrays.CuArray{T, 1}
    nthreads :: Int
    nblocks :: Int
end


function Guard(unit::Units.UnitR, grid::Grids.GridR, w0::Float64,
               medium::Media.Medium, rguard::Float64, kguard::Float64)
    # Spatial guard filter:
    Rguard = guard_window(grid.r, rguard, mode="right")
    Rguard = CuArrays.cu(convert(Array{FloatGPU, 1}, Rguard))

    # Angular guard filter:
    k0 = Media.k_func.(Ref(medium), w0)
    kmax = k0 * sind(kguard)
    Kguard = @. exp(-((grid.k * unit.k)^2 / kmax^2)^20)
    Kguard = CuArrays.cu(convert(Array{FloatGPU, 1}, Kguard))

    return GuardR(Rguard, Kguard)
end


function Guard(unit::Units.UnitRT, grid::Grids.GridRT, medium::Media.Medium,
               rguard::Float64, tguard::Float64, kguard::Float64,
               wguard::Float64)
    # Spatial guard filter:
    Rguard = guard_window(grid.r, rguard, mode="right")
    Rguard = CuArrays.cu(convert(Array{FloatGPU, 1}, Rguard))

    # Temporal guard filter:
    Tguard = guard_window(grid.t, tguard, mode="both")
    Tguard = CuArrays.cu(convert(Array{FloatGPU, 1}, Tguard))

    # Frequency guard filter:
    Wguard = @. exp(-((grid.w * unit.w)^2 / wguard^2)^20)
    Wguard = CuArrays.cu(convert(Array{FloatGPU, 1}, Wguard))

    # Angular guard filter:
    k = Media.k_func.(Ref(medium), grid.w * unit.w)
    kmax = k * sind(kguard)
    Kguard = zeros((grid.Nr, grid.Nw))
    for j=2:grid.Nw   # from 2 because kmax[1]=0 since w[1]=0
        for i=1:grid.Nr
            if kmax[j] != 0.
                Kguard[i, j] = exp(-((grid.k[i] * unit.k)^2 / kmax[j]^2)^20)
            end
        end
    end
    Kguard = CuArrays.cu(convert(Array{FloatGPU, 2}, Kguard))

    # GPU:
    CuArrays.allowscalar(false)   # disable slow fallback methods

    dev = CUDAnative.CuDevice(0)
    MAX_THREADS_PER_BLOCK = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
    nthreadsNt = min(grid.Nt, MAX_THREADS_PER_BLOCK)
    nthreadsNrNt = min(grid.Nr * grid.Nt, MAX_THREADS_PER_BLOCK)
    nthreadsNrNw = min(grid.Nr * grid.Nw, MAX_THREADS_PER_BLOCK)
    nblocksNt = Int(ceil(grid.Nt / nthreadsNt))
    nblocksNrNt = Int(ceil(grid.Nr * grid.Nt / nthreadsNrNt))
    nblocksNrNw = Int(ceil(grid.Nr * grid.Nw / nthreadsNrNw))

    return GuardRT(Rguard, Kguard, Tguard, Wguard,
                   nthreadsNt, nthreadsNrNt, nthreadsNrNw,
                   nblocksNt, nblocksNrNt, nblocksNrNw)
end


function Guard(unit::Units.UnitXY, grid::Grids.GridXY, w0::Float64,
               medium::Media.Medium, xguard::Float64, yguard::Float64,
               kxguard::Float64, kyguard::Float64)
    # Spatial guard filters:
    Xguard = guard_window(grid.x, xguard, mode="both")
    Xguard = CuArrays.cu(convert(Array{FloatGPU, 1}, Xguard))

    Yguard = guard_window(grid.y, yguard, mode="both")
    Yguard = CuArrays.cu(convert(Array{FloatGPU, 1}, Yguard))

    # Angular guard filters:
    k0 = Media.k_func.(Ref(medium), w0)
    kxmax = k0 * sind(kxguard)
    kymax = k0 * sind(kyguard)

    KXguard = @. exp(-((grid.kx * unit.kx)^2 / kxmax^2)^20)
    KXguard = CuArrays.cu(convert(Array{FloatGPU, 1}, KXguard))

    KYguard = @. exp(-((grid.ky * unit.ky)^2 / kymax^2)^20)
    KYguard = CuArrays.cu(convert(Array{FloatGPU, 1}, KYguard))

    dev = CUDAnative.CuDevice(0)
    MAX_THREADS_PER_BLOCK = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
    nthreads = min(grid.Nx * grid.Ny, MAX_THREADS_PER_BLOCK)
    nblocks = Int(ceil(grid.Nx * grid.Ny / nthreads))

    return GuardXY(Xguard, Yguard, KXguard, KYguard, nthreads, nblocks)
end


"""
Lossy guard window at the ends of grid coordinate.

    x: grid coordinate
    guard_width: the width of the lossy guard window
    mode: "left" - lossy guard only on the left end of the grid
          "right" - lossy guard only on the right end of the grid
          "both" - lossy guard on both ends of the grid
"""
function guard_window(x::Array{Float64, 1}, guard_width::Float64; mode="both")
    @assert guard_width >= 0.
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
                gauss1[i] = 1. - exp(-((x[i] - guard_xmin) / width)^6)
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
                gauss2[i] = 1. - exp(-((x[i] - guard_xmax) / width)^6)
            end
        end
        guard_right = 0.5 * (gauss1 + gauss2)

        # Result guard:
        if mode == "left"
            guard = guard_left
        elseif mode == "right"
            guard = guard_right
        elseif mode == "both"
            guard = @. guard_left + guard_right - 1.
        end
    end
    return guard
end


function apply_spatial_filter!(guard::GuardR,
                               E::CuArrays.CuArray{Complex{T}, 1}) where T
    @. E = E * guard.R
    return nothing
end


function apply_spatial_filter!(guard::GuardXY,
                               E::CuArrays.CuArray{Complex{T}, 2}) where T
    nth = guard.nthreads
    nbl = guard.nblocks
    @CUDAnative.cuda blocks=nbl threads=nth apply_spatial_filter_kernel(E, guard.X, guard.Y)
    return nothing
end


function apply_spatial_filter_kernel(E, X, Y)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    Nx, Ny = size(E)
    cartesian = CartesianIndices((Nx, Ny))
    for k=id:stride:Nx*Ny
        i = cartesian[k][1]
        j = cartesian[k][2]
        @inbounds E[i, j] = E[i, j] * X[i] * Y[j]
    end
    return nothing
end


function apply_angular_filter!(guard::GuardR,
                               E::CuArrays.CuArray{Complex{T}, 1}) where T
    @. E = E * guard.K
    return nothing
end


function apply_angular_filter!(guard::GuardXY,
                               E::CuArrays.CuArray{Complex{T}, 2}) where T
    nth = guard.nthreads
    nbl = guard.nblocks
    @CUDAnative.cuda blocks=nbl threads=nth apply_spatial_filter_kernel(E, guard.KX, guard.KY)
    return nothing
end


function apply_spatio_temporal_filter!(guard::GuardRT,
                                       E::CuArrays.CuArray{T, 2}) where T
    nth = guard.nthreadsNrNt
    nbl = guard.nblocksNrNt
    @CUDAnative.cuda blocks=nbl threads=nth apply_spatio_temporal_filter_kernel(E, guard.R, guard.T)
end


function apply_spatio_temporal_filter_kernel(E, R, T)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    Nr, Nt = size(E)
    cartesian = CartesianIndices((Nr, Nt))
    for k=id:stride:Nr*Nt
        i = cartesian[k][1]
        j = cartesian[k][2]
        @inbounds E[i, j] = E[i, j] * R[i] * T[j]
    end
    return nothing
end


function apply_frequency_angular_filter!(guard::GuardRT,
                                         S::CuArrays.CuArray{Complex{T}, 2}) where T
    nth = guard.nthreadsNrNw
    nbl = guard.nblocksNrNw
    @CUDAnative.cuda blocks=nbl threads=nth apply_frequency_angular_filter_kernel(S, guard.W, guard.K)
end


function apply_frequency_angular_filter_kernel(S, W, K)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    Nr, Nw = size(S)
    cartesian = CartesianIndices((Nr, Nw))
    for k=id:stride:Nr*Nw
        i = cartesian[k][1]
        j = cartesian[k][2]
        @inbounds S[i, j] = S[i, j] * W[j] * K[i, j]
    end
    return nothing
end


end

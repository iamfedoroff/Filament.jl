module Guards

import CUDAnative
import CuArrays
import CUDAdrv

import Units
import Grids
import Media

const FloatGPU = Float32
const ComplexGPU = ComplexF32


struct GuardFilter
    R :: CuArrays.CuArray{FloatGPU, 1}
    T :: CuArrays.CuArray{FloatGPU, 1}
    W :: CuArrays.CuArray{FloatGPU, 1}
    K :: CuArrays.CuArray{FloatGPU, 2}
    nthreadsNt :: Int64
    nthreadsNrNt :: Int64
    nthreadsNrNw :: Int64
    nblocksNt :: Int64
    nblocksNrNt :: Int64
    nblocksNrNw :: Int64
end


function GuardFilter(unit::Units.Unit, grid::Grids.Grid, medium::Media.Medium,
                     rguard_width::Float64, tguard_width::Float64,
                     kguard::Float64, wguard::Float64)
    # Spatial guard filter:
    Rguard = guard_window(grid.r, rguard_width, mode="right")
    Rguard = CuArrays.cu(convert(Array{FloatGPU, 1}, Rguard))

    # Temporal guard filter:
    Tguard = guard_window(grid.t, tguard_width, mode="both")
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

    return GuardFilter(Rguard, Tguard, Wguard, Kguard,
                       nthreadsNt, nthreadsNrNt, nthreadsNrNw,
                       nblocksNt, nblocksNrNt, nblocksNrNw)
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


function apply_temporal_filter!(guard::GuardFilter,
                                E::CuArrays.CuArray{FloatGPU, 1})
    nth = guard.nthreadsNt
    nbl = guard.nblocksNt
    @CUDAnative.cuda blocks=nbl threads=nth apply_temporal_filter_kernel(E, guard.T)
end


function apply_temporal_filter_kernel(E, T)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    N = length(E)
    for i=id:stride:N
        @inbounds E[i] = E[i] * T[i]
    end
    return nothing
end


function apply_spatio_temporal_filter!(guard::GuardFilter,
                                       E::CuArrays.CuArray{ComplexGPU, 2})
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


function apply_frequency_angular_filter!(guard::GuardFilter,
                                         S::CuArrays.CuArray{ComplexGPU, 2})
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

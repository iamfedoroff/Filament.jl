module Guards

import CuArrays
import CUDAnative
import CUDAdrv

import Units
import Grids
import Media

const FloatGPU = Float32
const ComplexGPU = Complex64


struct GuardFilter
    R :: Array{Float64, 1}
    T :: Array{Float64, 1}
    K :: Array{Float64, 2}
    W :: Array{Float64, 1}
    threadsNr :: Int64
    threadsNt :: Int64
    threadsNw :: Int64
    blocksNr :: Int64
    blocksNt :: Int64
    blocksNw :: Int64
    R_gpu :: CuArrays.CuArray{FloatGPU, 1}
    T_gpu :: CuArrays.CuArray{FloatGPU, 1}
    K_gpu :: CuArrays.CuArray{FloatGPU, 2}
    W_gpu :: CuArrays.CuArray{FloatGPU, 1}
end


function GuardFilter(unit::Units.Unit, grid::Grids.Grid, medium::Media.Medium,
                     rguard_width::Float64, tguard_width::Float64,
                     kguard::Float64, wguard::Float64)
    # Spatial guard filter:
    Rguard = guard_window(grid.r, rguard_width, mode="right")

    # Temporal guard filter:
    Tguard = guard_window(grid.t, tguard_width, mode="both")

    # Angular guard filter:
    k = Media.k_func.(medium, grid.w * unit.w)
    kmax = k * sind(kguard)
    Kguard = zeros((grid.Nr, grid.Nw))
    for j=2:grid.Nw   # from 2 because kmax[1]=0 since w[1]=0
        for i=1:grid.Nr
            if kmax[j] != 0.
                Kguard[i, j] = exp(-((grid.k[i] * unit.k)^2 / kmax[j]^2)^20)
            end
        end
    end

    # Spectral guard filter:
    Wguard = @. exp(-((grid.w * unit.w)^2 / wguard^2)^20)

    # GPU:
    dev = CUDAnative.CuDevice(0)
    MAX_THREADS = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
    threadsNr = min(grid.Nr, MAX_THREADS)
    threadsNt = min(grid.Nt, MAX_THREADS)
    threadsNw = min(grid.Nw, MAX_THREADS)
    blocksNr = Int(ceil(grid.Nr / threadsNr))
    blocksNt = Int(ceil(grid.Nt / threadsNt))
    blocksNw = Int(ceil(grid.Nw / threadsNw))

    Rguard_gpu = CuArrays.cu(convert(Array{FloatGPU, 1}, Rguard))
    Tguard_gpu = CuArrays.cu(convert(Array{FloatGPU, 1}, Tguard))
    Kguard_gpu = CuArrays.cu(convert(Array{FloatGPU, 2}, Kguard))
    Wguard_gpu = CuArrays.cu(convert(Array{FloatGPU, 1}, Wguard))

    return GuardFilter(Rguard, Tguard, Kguard, Wguard,
                       threadsNr, threadsNt, threadsNw,
                       blocksNr, blocksNt, blocksNw,
                       Rguard_gpu, Tguard_gpu, Kguard_gpu, Wguard_gpu)
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
    assert(guard_width >= 0.)
    assert(mode in ["left", "right", "both"])

    if mode in ["left", "right"]
        assert(guard_width <= x[end] - x[1])
    else
        assert(guard_width <= 0.5 * (x[end] - x[1]))
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
            guard = guard_left + guard_right - 1.
        end
    end
    return guard
end


function apply_spatial_filter!(guard::GuardFilter,
                               E_gpu::CuArrays.CuArray{ComplexGPU, 2})
    @CUDAnative.cuda blocks=guard.blocksNr threads=guard.threadsNr kernel1(E_gpu, guard.R_gpu)
end


function apply_temporal_filter!(guard::GuardFilter,
                                E_gpu::CuArrays.CuArray{FloatGPU, 1})
    @CUDAnative.cuda blocks=guard.blocksNt threads=guard.threadsNt kernel0(E_gpu, guard.T_gpu)
end


function apply_temporal_filter!(guard::GuardFilter,
                                E_gpu::CuArrays.CuArray{ComplexGPU, 2})
    @CUDAnative.cuda blocks=guard.blocksNt threads=guard.threadsNt kernel2(E_gpu, guard.T_gpu)
end


function apply_spectral_filter!(guard::GuardFilter,
                                S_gpu::CuArrays.CuArray{ComplexGPU, 2})
    @CUDAnative.cuda blocks=guard.blocksNw threads=guard.threadsNw kernel2(S_gpu, guard.W_gpu)
end


function apply_angular_filter!(guard::GuardFilter,
                               S_gpu::CuArrays.CuArray{ComplexGPU, 2})
    @CUDAnative.cuda blocks=guard.blocksNw threads=guard.threadsNw kernel3(S_gpu, guard.K_gpu)
end


function kernel0(a, b)
    i = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    if i <= length(a)
        a[i] = a[i] * b[i]
    end
    return nothing
end


function kernel1(a, b)
    N1, N2 = size(a)
    i = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    if i <= N1
        for j=1:N2
            a[i, j] = a[i, j] * b[i]
        end
    end
    return nothing
end


function kernel2(a, b)
    N1, N2 = size(a)
    j = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    if j <= N2
        for i=1:N1
            a[i, j] = a[i, j] * b[j]
        end
    end
    return nothing
end


function kernel3(a, b)
    N1, N2 = size(a)
    # Since usually N2 > N1, I choose to parallel along the second dimension:
    j = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    if j <= N2
        for i=1:N1
            a[i, j] = a[i, j] * b[i, j]
        end
    end

    return nothing
end


end

module AnalyticSignals

import CuArrays
import CUDAdrv
import CUDAnative
import FFTW

import ..FourierTransforms

const MAX_THREADS_PER_BLOCK = CUDAdrv.attribute(
    CUDAnative.CuDevice(0), CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
)


"""
Transforms the spectrum of a real signal to the spectrum of the corresponding
analytic signal:
            { 2 * Sr(f),  f > 0
    Sa(f) = { Sr(f),      f = 0   = [1 + sgn(f)] * Sr(f)
            { 0,          f < 0
"""
function rspec2aspec!(S::AbstractArray{Complex{T}, 1}) where T
    N = length(S)
    Nhalf = half(N)
    # S[1] = S[1]   # f = 0
    for i=2:Nhalf
        S[i] = 2 * S[i]   # f > 0
    end
    for i=Nhalf+1:N
        S[i] = 0   # f < 0
    end
    return nothing
end


function rspec2aspec!(S::CuArrays.CuArray{Complex{T}}) where T
    N = length(S)
    nth = min(N, MAX_THREADS_PER_BLOCK)
    nbl = cld(N, nth)
    @CUDAnative.cuda blocks=nbl threads=nth _rspec2aspec_kernel!(S)
    return nothing
end


function _rspec2aspec_kernel!(
    S::CUDAnative.CuDeviceArray{Complex{T}, 2},
) where T
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x +
         CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    Nr, Nt = size(S)
    Nthalf = half(Nt)
    cartesian = CartesianIndices((Nr, Nt))
    for k=id:stride:Nr*Nt
        ir = cartesian[k][1]
        it = cartesian[k][2]

        if (it >= 2) & (it <= Nthalf)
            S[ir, it] = 2 * S[ir, it]
        end
        if (it >= Nthalf + 1) & (it <= Nt)
            S[ir, it] = 0
        end
    end
    return nothing
end


function _rspec2aspec_kernel!(
    S::CUDAnative.CuDeviceArray{Complex{T}, 3},
) where T
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x +
         CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    Nx, Ny, Nt = size(S)
    Nthalf = half(Nt)
    cartesian = CartesianIndices((Nx, Ny, Nt))
    for k=id:stride:Nx*Ny*Nt
        ix = cartesian[k][1]
        iy = cartesian[k][2]
        it = cartesian[k][3]

        if (it >= 2) & (it <= Nthalf)
            S[ix, iy, it] = 2 * S[ix, iy, it]
        end
        if (it >= Nthalf + 1) & (it <= Nt)
            S[ix, iy, it] = 0
        end
    end
    return nothing
end


"""
Transforms the spectrum of an analytic signal to the spectrum of the
corresponding real signal:
            { Sa(f) / 2,         f > 0
    Sr(f) = { Sa(f),             f = 0   = [Sa(f) + conj(Sa(-f))] / 2
            { conj(Sa(-f)) / 2,  f < 0
"""
function aspec2rspec!(S::AbstractArray{Complex{T}, 1}) where T
    N = length(S)
    Nhalf = half(N)
    # S[1] = S[1]   # f = 0
    for i=2:Nhalf
        S[i] = S[i] / 2   # f > 0
    end
    for i=Nhalf+1:N
        S[i] = conj(S[N - i + 2])   # f < 0
    end
    return nothing
end


function aspec2rspec!(
    Sr::AbstractArray{Complex{T}},
    Sa::AbstractArray{Complex{T}},
) where T
    N = length(Sr)
    Sr[1] = Sa[1]   # f = 0
    for i=2:N
        Sr[i] = Sa[i] / 2   # f > 0
    end
    return nothing
end


function aspec2rspec!(
    Sr::CuArrays.CuArray{Complex{T}},
    Sa::CuArrays.CuArray{Complex{T}},
) where T
    N = length(Sr)
    nth = min(N, MAX_THREADS_PER_BLOCK)
    nbl = cld(N, nth)
    @CUDAnative.cuda blocks=nbl threads=nth _aspec2rspec_kernel!(Sr, Sa)
    return nothing
end


function _aspec2rspec_kernel!(
    Sr::CUDAnative.CuDeviceArray{Complex{T}, 2},
    Sa::CUDAnative.CuDeviceArray{Complex{T}, 2},
) where T
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x +
         CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    Nr, Nw = size(Sr)
    cartesian = CartesianIndices((Nr, Nw))
    for k=id:stride:Nr*Nw
        i = cartesian[k][1]
        j = cartesian[k][2]
        if j == 1
            Sr[i, j] = Sa[i, j]
        else
            Sr[i, j] = Sa[i, j] / 2
        end
    end
    return nothing
end


"""
Transforms a real signal to the corresponding analytic signal.
"""
function rsig2asig!(
    E::AbstractArray{Complex{T}},
    FT::FourierTransforms.Plan,
) where T
    FourierTransforms.fft!(E, FT)
    rspec2aspec!(E)
    FourierTransforms.ifft!(E, FT)
    return nothing
end


"""
Transforms an analytic signal to the corresponding real signal.
"""
function asig2rsig!(E::AbstractArray{Complex{T}}) where T
    @. E = real(E)
    return nothing
end


"""
Transforms a real signal to the spectrum of the corresponding analytic signal.
"""
function rsig2aspec!(
    E::AbstractArray{Complex{T}},
    FT::FourierTransforms.Plan,
) where T
    FourierTransforms.fft!(E, FT)
    rspec2aspec!(E)
    return nothing
end


function half(N::Int)
    if iseven(N)
        Nhalf = div(N, 2)
    else
        Nhalf = div(N + 1, 2)
    end
    return Nhalf
end


end

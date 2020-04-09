module AnalyticSignals

import CuArrays
import CUDAdrv
import CUDAnative
import FFTW

import Constants: MAX_THREADS_PER_BLOCK
import FourierTransforms


"""
Transforms the spectrum of a real signal to the spectrum of the corresponding
analytic signal.
"""
function rspec2aspec!(S::AbstractArray{Complex{T}, 1}) where T
    N = length(S)
    Nhalf = half(N)
    for i=2:Nhalf
        S[i] = convert(T, 2) * S[i]
    end
    for i=Nhalf+1:N
        S[i] = convert(T, 0)
    end
    return nothing
end


function rspec2aspec!(S::CuArrays.CuArray{Complex{T}, 2}) where T
    N = length(S)
    nth = min(N, MAX_THREADS_PER_BLOCK)
    nbl = Int(ceil(N / nth))
    @CUDAnative.cuda blocks=nbl threads=nth _rspec2aspec_kernel!(S)
    return nothing
end


function rspec2aspec!(S::CuArrays.CuArray{Complex{T}, 3}) where T
    N = length(S)
    nth = min(N, MAX_THREADS_PER_BLOCK)
    nbl = Int(ceil(N / nth))
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
            S[ir, it] = convert(T, 2) * S[ir, it]
        end
        if (it >= Nthalf + 1) & (it <= Nt)
            S[ir, it] = convert(T, 0)
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
            S[ix, iy, it] = convert(T, 2) * S[ix, iy, it]
        end
        if (it >= Nthalf + 1) & (it <= Nt)
            S[ix, iy, it] = convert(T, 0)
        end
    end
    return nothing
end


"""
Transforms the spectrum of an analytic signal to the spectrum of the
corresponding real signal.
"""
function aspec2rspec!(S::AbstractArray{Complex{T}, 1}) where T
    N = length(S)
    Nhalf = half(N)
    for i=2:Nhalf
        S[i] = convert(T, 0.5) * S[i]
    end
    for i=Nhalf+1:N-1
        S[i] = S[N - i + 2]
    end
    return nothing
end


"""
Transforms a real signal to the corresponding analytic signal.
"""
function rsig2asig!(
    E::AbstractArray{Complex{T}},
    FT::FourierTransforms.FourierTransform,
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
    FT::FourierTransforms.FourierTransform,
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

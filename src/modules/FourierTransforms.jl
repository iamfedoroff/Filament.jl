module FourierTransforms

import FFTW
import CUDAnative
import CuArrays
import CUDAdrv

import Constants: FloatGPU, MAX_THREADS_PER_BLOCK


abstract type FourierTransform end


struct FourierTransformT <: FourierTransform
    Nt :: Int
    p_fft! :: FFTW.Plan
    p_ifft! :: FFTW.Plan
end


struct FourierTransformRT <: FourierTransform
    Nr :: Int
    Nt :: Int
    p_fft! :: FFTW.Plan
    p_ifft! :: FFTW.Plan
end


struct FourierTransformXY <: FourierTransform
    Nx :: Int
    Ny :: Int
    p_fft! :: FFTW.Plan
    p_ifft! :: FFTW.Plan
end


struct FourierTransformXYT <: FourierTransform
    Nx :: Int
    Ny :: Int
    Nt :: Int
    p_fft! :: FFTW.Plan
    p_ifft! :: FFTW.Plan
    p_fft2! :: FFTW.Plan
    p_ifft2! :: FFTW.Plan
end


function FourierTransformT(Nt::Int)
    Ftmp = zeros(ComplexF64, Nt)
    p_fft! = FFTW.plan_fft!(Ftmp)
    p_ifft! = FFTW.plan_ifft!(Ftmp)
    return FourierTransformT(Nt, p_fft!, p_ifft!)
end


function FourierTransformRT(Nr::T, Nt::T) where T<:Int
    CuArrays.allowscalar(false)   # disable slow fallback methods
    Ftmp = CuArrays.zeros(Complex{FloatGPU}, (Nr, Nt))
    p_fft! = FFTW.plan_fft(Ftmp, [2])
    p_ifft! = FFTW.plan_ifft(Ftmp, [2])
    return FourierTransformRT(Nr, Nt, p_fft!, p_ifft!)
end


function FourierTransformXY(Nx::T, Ny::T) where T<:Int
    CuArrays.allowscalar(false)   # disable slow fallback methods
    p_fft! = FFTW.plan_fft(CuArrays.zeros(Complex{FloatGPU}, (Nx, Ny)))
    p_ifft! = FFTW.plan_ifft(CuArrays.zeros(Complex{FloatGPU}, (Nx, Ny)))
    return FourierTransformXY(Nx, Ny, p_fft!, p_ifft!)
end


function FourierTransformXYT(Nx::T, Ny::T, Nt::T) where T<:Int
    CuArrays.allowscalar(false)   # disable slow fallback methods
    Ftmp = CuArrays.zeros(Complex{FloatGPU}, (Nx, Ny, Nt))
    p_fft! = FFTW.plan_fft(Ftmp, [3])
    p_ifft! = FFTW.plan_ifft(Ftmp, [3])
    p_fft2! = FFTW.plan_fft(Ftmp, [1, 2])
    p_ifft2! = FFTW.plan_ifft(Ftmp, [1, 2])
    return FourierTransformXYT(Nx, Ny, Nt, p_fft!, p_ifft!, p_fft2!, p_ifft2!)
end


function fft!(
    E::AbstractArray{Complex{T}},
    FT::FourierTransform,
) where T
    FFTW.mul!(E, FT.p_fft!, E)
    return nothing
end


function fft2!(
    E::AbstractArray{Complex{T}},
    FT::FourierTransform,
) where T
    FFTW.mul!(E, FT.p_fft2!, E)
    return nothing
end


function ifft!(
    E::AbstractArray{Complex{T}},
    FT::FourierTransform,
) where T
    FFTW.mul!(E, FT.p_ifft!, E)
    return nothing
end


function ifft2!(
    E::AbstractArray{Complex{T}},
    FT::FourierTransform,
) where T
    FFTW.mul!(E, FT.p_ifft2!, E)
    return nothing
end


function convolution!(
    x::AbstractArray{Complex{T}, 1},
    FT::FourierTransform,
    H::AbstractArray{Complex{T}, 1},
) where T
    fft!(x, FT)
    @. x = H * x
    ifft!(x, FT)
    return nothing
end


function convolution!(
    x::CuArrays.CuArray{Complex{T}, 2},
    FT::FourierTransform,
    H::CuArrays.CuArray{Complex{T}, 1},
) where T
    fft!(x, FT)
    N = length(x)
    nth = min(N, MAX_THREADS_PER_BLOCK)
    nbl = Int(ceil(N / nth))
    @CUDAnative.cuda blocks=nbl threads=nth _convolution_kernel!(x, H)
    ifft!(x, FT)
    return nothing
end


function _convolution_kernel!(x, H)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x +
         CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    Nr, Nt = size(x)
    cartesian = CartesianIndices((Nr, Nt))
    for k=id:stride:Nr*Nt
        i = cartesian[k][1]
        j = cartesian[k][2]
        x[i, j] = H[j] * x[i, j]
    end
    return nothing
end


"""
Return the Discrete Fourier Transform sample frequencies.
https://github.com/numpy/numpy/blob/v1.15.0/numpy/fft/helper.py#L124-L169
"""
function fftfreq(n::Int, d::T) where T<:AbstractFloat
    val = 1 / (n * d)
    results = zeros(T, n)
    N = Int(floor((n - 1) / 2)) + 1
    p1 = Array(0:N-1)
    results[1:N] = p1
    p2 = Array(-Int(floor(n / 2)):-1)
    results[N+1:end] = p2
    return results * val
end


"""
Return the Discrete Fourier Transform sample frequencies (for usage with
rfft, irfft).
https://github.com/numpy/numpy/blob/v1.15.0/numpy/fft/helper.py#L173-L221
"""
function rfftfreq(n::Int, d::T) where T<:AbstractFloat
    val = 1 / (n * d)
    N = Int(floor(n / 2))
    results = Array{T}(0:N)
    return results * val
end


function rfft_length(Nt::Int)
    if iseven(Nt)
        Nw = div(Nt, 2) + 1
    else
        Nw = div(Nt + 1, 2)
    end
    return Nw
end


"""
Circular-shift along the given dimension of a periodic signal 'x' centered at
index '1' so it becomes centered at index 'N / 2 + 1', where 'N' is the size of
that dimension.
https://github.com/JuliaMath/AbstractFFTs.jl/blob/66695a72b2a29a059a9bf5fae51a3107172f146d/src/definitions.jl#L349
"""
fftshift(x) = circshift(x, div.([size(x)...],2))


"""
Circular-shift along the given dimension of a periodic signal 'x' centered at
index 'N / 2 + 1' so it becomes centered at index '1', where 'N' is the size of
that dimension.
https://github.com/JuliaMath/AbstractFFTs.jl/blob/66695a72b2a29a059a9bf5fae51a3107172f146d/src/definitions.jl#L374
"""
ifftshift(x) = circshift(x, div.([size(x)...],-2))


end

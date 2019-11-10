module Fourier

import FFTW
import CUDAnative
import CuArrays
import CUDAdrv

const FloatGPU = Float32
const MAX_THREADS_PER_BLOCK =
        CUDAdrv.attribute(
            CUDAnative.CuDevice(0),
            CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        )


abstract type FourierTransform end


struct FourierTransformT{T} <: FourierTransform
    Nt :: Int
    Nw :: Int

    Er :: AbstractArray{T, 1}
    Sr :: AbstractArray{Complex{T}, 1}
    Sc :: AbstractArray{Complex{T}, 1}

    p_rfft :: FFTW.Plan
    p_irfft :: FFTW.Plan
    p_ifft :: FFTW.Plan
end


struct FourierTransformRT{T} <: FourierTransform
    Nr :: Int
    Nt :: Int
    Nw :: Int

    Er2 :: CuArrays.CuArray{T, 2}
    Sr :: CuArrays.CuArray{Complex{T}, 1}
    Sr2 :: CuArrays.CuArray{Complex{T}, 2}
    Sc :: CuArrays.CuArray{Complex{T}, 1}
    Sc2 :: CuArrays.CuArray{Complex{T}, 2}

    p_rfft :: FFTW.Plan
    p_rfft2 :: FFTW.Plan
    p_irfft :: FFTW.Plan
    p_irfft2 :: FFTW.Plan
    p_ifft :: FFTW.Plan
    p_ifft2 :: FFTW.Plan
end


struct FourierTransformXY <: FourierTransform
    Nx :: Int
    Ny :: Int

    p_fft2 :: FFTW.Plan
    p_ifft2 :: FFTW.Plan
end


function FourierTransformT(Nt::T) where T<:Int
    Nw = rfft_length(Nt)

    Er = zeros(Nt)
    Sr = zeros(ComplexF64, Nw)
    Sc = zeros(ComplexF64, Nt)

    p_rfft = FFTW.plan_rfft(Er)
    p_irfft = FFTW.plan_irfft(Sr, Nt)
    p_ifft = FFTW.plan_ifft(Sc)

    return FourierTransformT(Nt, Nw, Er, Sr, Sc, p_rfft, p_irfft, p_ifft)
end


function FourierTransformRT(Nr::T, Nt::T) where T<:Int
    CuArrays.allowscalar(false)   # disable slow fallback methods

    Nw = rfft_length(Nt)

    Er2 = CuArrays.zeros(FloatGPU, (Nr, Nt))
    Sr = CuArrays.zeros(Complex{FloatGPU}, Nw)
    Sr2 = CuArrays.zeros(Complex{FloatGPU}, (Nr, Nw))
    Sc = CuArrays.zeros(Complex{FloatGPU}, Nt)
    Sc2 = CuArrays.zeros(Complex{FloatGPU}, (Nr, Nt))

    p_rfft = FFTW.plan_rfft(CuArrays.zeros(FloatGPU, Nt))
    p_rfft2 = FFTW.plan_rfft(Er2, [2])
    p_irfft = FFTW.plan_irfft(Sr, Nt)
    p_irfft2 = FFTW.plan_irfft(Sr2, Nt, [2])
    p_ifft = FFTW.plan_ifft(Sc)
    p_ifft2 = FFTW.plan_ifft(Sc2, [2])

    return FourierTransformRT(
        Nr, Nt, Nw,
        Er2, Sr, Sr2, Sc, Sc2,
        p_rfft, p_rfft2, p_irfft, p_irfft2, p_ifft, p_ifft2,
    )
end


function FourierTransformXY(Nx::T, Ny::T) where T<:Int
    CuArrays.allowscalar(false)   # disable slow fallback methods

    p_fft2 = FFTW.plan_fft(CuArrays.zeros(Complex{FloatGPU}, (Nx, Ny)))
    p_ifft2 = FFTW.plan_ifft(CuArrays.zeros(Complex{FloatGPU}, (Nx, Ny)))

    return FourierTransformXY(Nx, Ny, p_fft2, p_ifft2)
end


function fft!(
    E::CuArrays.CuArray{Complex{T}, 2},
    FT::FourierTransform,
) where T
    FFTW.mul!(E, FT.p_fft2, E)   # space -> frequency
    return nothing
end


function ifft!(
    E::CuArrays.CuArray{Complex{T}, 2},
    FT::FourierTransform,
) where T
    FFTW.mul!(E, FT.p_ifft2, E)   # space -> frequency
    return nothing
end


function ifft!(
    E::AbstractArray{Complex{T}, 1},
    FT::FourierTransform,
    S::AbstractArray{Complex{T}, 1},
) where T
    FFTW.mul!(E, FT.p_ifft, S)   # frequency -> time
    return nothing
end


function ifft!(
    E::CuArrays.CuArray{Complex{T}, 2},
    FT::FourierTransform,
    S::CuArrays.CuArray{Complex{T}, 2},
) where T
    FFTW.mul!(E, FT.p_ifft2, S)   # frequency -> time
    return nothing
end


function rfft!(
    S::AbstractArray{Complex{T}, 1},
    FT::FourierTransform,
    E::AbstractArray{T, 1},
) where T
    FFTW.mul!(S, FT.p_rfft, E)   # time -> frequency
    return nothing
end


function rfft!(
    S::AbstractArray{Complex{T}, 1},
    FT::FourierTransform,
    E::AbstractArray{Complex{T}, 1},
) where T
    @. FT.Er = real(E)
    FFTW.mul!(S, FT.p_rfft, FT.Er)   # time -> frequency
    return nothing
end


function rfft!(
    S::CuArrays.CuArray{Complex{T}, 2},
    FT::FourierTransform,
    E::CuArrays.CuArray{T, 2},
) where T
    FFTW.mul!(S, FT.p_rfft2, E)   # time -> frequency
    return nothing
end


function rfft!(
    S::CuArrays.CuArray{Complex{T}, 2},
    FT::FourierTransform,
    E::CuArrays.CuArray{Complex{T}, 2},
) where T
    N = length(E)
    nth = min(N, MAX_THREADS_PER_BLOCK)
    nbl = Int(ceil(N / nth))
    @CUDAnative.cuda blocks=nbl threads=nth _rfft_kernel!(FT.Er2, E)
    FFTW.mul!(S, FT.p_rfft2, FT.Er2)   # time -> frequency
    return nothing
end


function _rfft_kernel!(Er, Ec)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x +
         CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    N1, N2 = size(Ec)
    cartesian = CartesianIndices((N1, N2))
    for k=id:stride:N1*N2
        i = cartesian[k][1]
        j = cartesian[k][2]
        Er[i, j] = real(Ec[i, j])
    end
    return nothing
end


function irfft!(
    E::AbstractArray{T, 1},
    FT::FourierTransform,
    S::AbstractArray{Complex{T}, 1},
) where T
    FFTW.mul!(E, FT.p_irfft, S)   # frequency -> time
    return nothing
end


function irfft!(
    E::CuArrays.CuArray{T, 2},
    FT::FourierTransform,
    S::CuArrays.CuArray{Complex{T}, 2},
) where T
    FFTW.mul!(E, FT.p_irfft2, S)   # frequency -> time
    return nothing
end


function hilbert!(
    Ec::AbstractArray{Complex{T}, 1},
    FT::FourierTransform,
    Sr::AbstractArray{Complex{T}, 1},
) where T
    for i=1:FT.Nt
        if i <= FT.Nw
            if i == 1
                FT.Sc[i] = Sr[i]
            else
                FT.Sc[i] = 2. * Sr[i]
            end
        else
            FT.Sc[i] = 0.
        end
    end
    ifft!(Ec, FT, FT.Sc)   # frequency -> time
    return nothing
end


"""
Transforms the spectruum of a real signal into the complex analytic signal.

WARNING: Needs test for odd N and low frequencies.
"""
function hilbert!(
    Ec::CuArrays.CuArray{Complex{T}, 1},
    FT::FourierTransform,
    Sr::CuArrays.CuArray{Complex{T}, 1},
) where T
    Nt = length(Ec)
    nth = min(Nt, MAX_THREADS_PER_BLOCK)
    nbl = Int(ceil(Nt / nth))
    @CUDAnative.cuda blocks=nbl threads=nth _hilbert_kernel!(FT.Sc, Sr)
    ifft!(Ec, FT, FT.Sc)   # frequency -> time
    return nothing
end


function _hilbert_kernel!(
    Sc::CUDAnative.CuDeviceArray{Complex{T}, 1},
    Sr::CUDAnative.CuDeviceArray{Complex{T}, 1},
) where T
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x +
         CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    Nt = length(Sc)
    Nw = length(Sr)
    for i=id:stride:Nt
        if i <= Nw
            if i == 1
                Sc[i] = Sr[i]
            else
                Sc[i] = FloatGPU(2) * Sr[i]
            end
        else
            Sc[i] = FloatGPU(0)
        end
    end
    return nothing
end


"""
Transforms the spectruum of a real signal into the complex analytic signal.

WARNING: Needs test for odd N and low frequencies.
"""
function hilbert!(
    Ec::CuArrays.CuArray{Complex{T}, 2},
    FT::FourierTransform,
    Sr::CuArrays.CuArray{Complex{T}, 2},
) where T
    N = length(Ec)
    nth = min(N, MAX_THREADS_PER_BLOCK)
    nbl = Int(ceil(N / nth))
    @CUDAnative.cuda blocks=nbl threads=nth _hilbert_kernel!(FT.Sc2, Sr)
    ifft!(Ec, FT, FT.Sc2)   # frequency -> time
    return nothing
end


function _hilbert_kernel!(
    Sc::CUDAnative.CuDeviceArray{Complex{T}, 2},
    Sr::CUDAnative.CuDeviceArray{Complex{T}, 2},
) where T
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x +
         CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    Nr, Nt = size(Sc)
    Nr, Nw = size(Sr)
    cartesian = CartesianIndices((Nr, Nt))
    for k=id:stride:Nr*Nt
        i = cartesian[k][1]
        j = cartesian[k][2]
        if j <= Nw
            if j == 1
                Sc[i, j] = Sr[i, j]
            else
                Sc[i, j] = FloatGPU(2) * Sr[i, j]
            end
        else
            Sc[i, j] = FloatGPU(0)
        end
    end
    return nothing
end


function convolution!(
    x::AbstractArray{T, 1},
    FT::FourierTransform,
    Hw::AbstractArray{Complex{T}, 1},
) where T
    rfft!(FT.Sr, FT, x)
    @. FT.Sr = Hw * FT.Sr
    irfft!(x, FT, FT.Sr)
    return nothing
end


function convolution!(
    x::CuArrays.CuArray{T, 2},
    FT::FourierTransform,
    Hw::CuArrays.CuArray{Complex{T}, 1},
) where T
    rfft!(FT.Sr2, FT, x)
    N = length(FT.Sr2)
    nth = min(N, MAX_THREADS_PER_BLOCK)
    nbl = Int(ceil(N / nth))
    @CUDAnative.cuda blocks=nbl threads=nth _convolution_kernel!(FT.Sr2, Hw)
    irfft!(x, FT, FT.Sr2)
    return nothing
end


function _convolution_kernel!(Sr, Hw)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x +
         CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    Nr, Nw = size(Sr)
    cartesian = CartesianIndices((Nr, Nw))
    for k=id:stride:Nr*Nw
        i = cartesian[k][1]
        j = cartesian[k][2]
        Sr[i, j] = Hw[j] * Sr[i, j]
    end
    return nothing
end


"""
Return the Discrete Fourier Transform sample frequencies.
https://github.com/numpy/numpy/blob/v1.15.0/numpy/fft/helper.py#L124-L169
"""
function fftfreq(n::Int, d::Float64)
    val = 1 / (n * d)
    results = zeros(n)
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
function rfftfreq(n::Int, d::Float64)
    val = 1 / (n * d)
    N = Int(floor(n / 2))
    results = Array(0:N)
    return results * val
end


"""
Undoes the effect of "fftshift", where "fftshift" swaps the first and second
halves of array "x".
https://github.com/JuliaMath/AbstractFFTs.jl/blob/master/src/definitions.jl#L387
"""
function ifftshift(x)
    return circshift(x, div.([size(x)...], -2))
end


function rfft_length(Nt::Int)
    if iseven(Nt)
        Nw = div(Nt, 2) + 1
    else
        Nw = div(Nt + 1, 2)
    end
    return Nw
end

end

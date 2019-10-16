module Fourier

import FFTW
import CUDAnative
import CuArrays
import CUDAdrv

const FloatGPU = Float32
const ComplexGPU = ComplexF32


abstract type FourierTransform end


struct FourierTransformT{T} <: FourierTransform
    Nt :: Int
    Nw :: Int

    Er :: AbstractArray{T, 1}
    Sc :: AbstractArray{Complex{T}, 1}
    Sr :: AbstractArray{Complex{T}, 1}

    prfft :: FFTW.Plan
    pirfft :: FFTW.Plan
    pifft :: FFTW.Plan
end


struct FourierTransformRT{T} <: FourierTransform
    Nr :: Int
    Nt :: Int
    Nw :: Int

    Er2 :: CuArrays.CuArray{T, 2}
    Sc :: CuArrays.CuArray{Complex{T}, 1}
    Sc2 :: CuArrays.CuArray{Complex{T}, 2}
    Sr :: CuArrays.CuArray{Complex{T}, 1}
    Sr2 :: CuArrays.CuArray{Complex{T}, 2}

    pifft :: FFTW.Plan
    pifft2 :: FFTW.Plan
    prfft :: FFTW.Plan
    prfft2 :: FFTW.Plan
    pirfft :: FFTW.Plan
    pirfft2 :: FFTW.Plan

    nthreadsNt :: Int
    nthreadsNw :: Int
    nblocksNt :: Int
    nblocksNw :: Int
    nthreadsNrNt :: Int
    nthreadsNrNw :: Int
    nblocksNrNt :: Int
    nblocksNrNw :: Int
end


struct FourierTransformXY <: FourierTransform
    Nx :: Int
    Ny :: Int

    pfft :: FFTW.Plan
    pifft :: FFTW.Plan
end


function FourierTransformT(Nt::T) where T<:Int
    if iseven(Nt)
        Nw = div(Nt, 2) + 1
    else
        Nw = div(Nt + 1, 2)
    end

    Er = zeros(Nt)
    Sc = zeros(ComplexF64, Nt)
    Sr = zeros(ComplexF64, Nw)

    prfft = FFTW.plan_rfft(zeros(Float64, Nt))
    pirfft = FFTW.plan_irfft(zeros(ComplexF64, Nw), Nt)
    pifft = FFTW.plan_ifft(zeros(ComplexF64, Nt))

    return FourierTransformT(Nt, Nw, Er, Sc, Sr, prfft, pirfft, pifft)
end


function FourierTransformRT(Nr::T, Nt::T) where T<:Int
    CuArrays.allowscalar(false)   # disable slow fallback methods

    if iseven(Nt)
        Nw = div(Nt, 2) + 1
    else
        Nw = div(Nt + 1, 2)
    end

    Er2 = CuArrays.zeros(FloatGPU, (Nr, Nt))
    Sc = CuArrays.zeros(ComplexGPU, Nt)
    Sc2 = CuArrays.zeros(ComplexGPU, (Nr, Nt))
    Sr = CuArrays.zeros(ComplexGPU, Nw)
    Sr2 = CuArrays.zeros(ComplexGPU, (Nr, Nw))

    pifft = FFTW.plan_ifft(CuArrays.zeros(ComplexGPU, Nt))
    pifft2 = FFTW.plan_ifft(CuArrays.zeros(ComplexGPU, (Nr, Nt)), [2])
    prfft = FFTW.plan_rfft(CuArrays.zeros(FloatGPU, Nt))
    prfft2 = FFTW.plan_rfft(CuArrays.zeros(FloatGPU, (Nr, Nt)), [2])
    pirfft = FFTW.plan_irfft(CuArrays.zeros(ComplexGPU, Nw), Nt)
    pirfft2 = FFTW.plan_irfft(CuArrays.zeros(ComplexGPU, (Nr, Nw)), Nt, [2])

    dev = CUDAnative.CuDevice(0)
    MAX_THREADS_PER_BLOCK = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
    nthreadsNt = min(Nt, MAX_THREADS_PER_BLOCK)
    nthreadsNw = min(Nw, MAX_THREADS_PER_BLOCK)
    nblocksNt = Int(ceil(Nt / nthreadsNt))
    nblocksNw = Int(ceil(Nw / nthreadsNw))
    nthreadsNrNt = min(Nr * Nt, MAX_THREADS_PER_BLOCK)
    nblocksNrNt = Int(ceil(Nr * Nt / nthreadsNrNt))
    nthreadsNrNw = min(Nr * Nw, MAX_THREADS_PER_BLOCK)
    nblocksNrNw = Int(ceil(Nr * Nw / nthreadsNrNt))

    return FourierTransformRT(
        Nr,
        Nt,
        Nw,
        Er2,
        Sc,
        Sc2,
        Sr,
        Sr2,
        pifft,
        pifft2,
        prfft,
        prfft2,
        pirfft,
        pirfft2,
        nthreadsNt,
        nthreadsNw,
        nblocksNt,
        nblocksNw,
        nthreadsNrNt,
        nthreadsNrNw,
        nblocksNrNt,
        nblocksNrNw,
    )
end


function FourierTransformXY(Nx::T, Ny::T) where T<:Int
    CuArrays.allowscalar(false)   # disable slow fallback methods

    pfft = FFTW.plan_fft(CuArrays.zeros(ComplexGPU, (Nx, Ny)))
    pifft = FFTW.plan_ifft(CuArrays.zeros(ComplexGPU, (Nx, Ny)))

    return FourierTransformXY(Nx, Ny, pfft, pifft)
end


function fft!(
    FT::FourierTransformXY,
    E::CuArrays.CuArray{Complex{T}, 2},
) where T
    FFTW.mul!(E, FT.pfft, E)   # space -> frequency
    return nothing
end


function ifft!(
    FT::FourierTransformXY,
    E::CuArrays.CuArray{Complex{T}, 2},
) where T
    FFTW.mul!(E, FT.pifft, E)   # space -> frequency
    return nothing
end


function ifft!(
    FT::FourierTransform,
    S::AbstractArray{Complex{T}, 1},
    E::AbstractArray{Complex{T}, 1},
) where T
    FFTW.mul!(E, FT.pifft, S)   # frequency -> time
    return nothing
end


function ifft2!(
    FT::FourierTransform,
    S::CuArrays.CuArray{Complex{T}, 2},
    E::CuArrays.CuArray{Complex{T}, 2},
) where T
    FFTW.mul!(E, FT.pifft2, S)   # frequency -> time
    return nothing
end


function rfft!(
    FT::FourierTransform,
    E::AbstractArray{T, 1},
    S::AbstractArray{Complex{T}, 1},
) where T
    FFTW.mul!(S, FT.prfft, E)   # time -> frequency
    return nothing
end


function rfft!(
    FT::FourierTransform,
    E::AbstractArray{Complex{T}, 1},
    S::AbstractArray{Complex{T}, 1},
) where T
    @. FT.Er = real(E)
    FFTW.mul!(S, FT.prfft, FT.Er)   # time -> frequency
    return nothing
end


function rfft2!(
    FT::FourierTransform,
    E::CuArrays.CuArray{T, 2},
    S::CuArrays.CuArray{Complex{T}, 2},
) where T
    FFTW.mul!(S, FT.prfft2, E)   # time -> frequency
    return nothing
end


function rfft2!(
    FT::FourierTransform,
    E::CuArrays.CuArray{Complex{T}, 2},
    S::CuArrays.CuArray{Complex{T}, 2},
) where T
    nth = FT.nthreadsNrNt
    nbl = FT.nblocksNrNt
    @CUDAnative.cuda blocks=nbl threads=nth rfft2_kernel(E, FT.Er2)
    FFTW.mul!(S, FT.prfft2, FT.Er2)   # time -> frequency
    return nothing
end


function rfft2_kernel(Ec, Er)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    Nr, Nt = size(Ec)
    cartesian = CartesianIndices((Nr, Nt))
    for k=id:stride:Nr*Nt
        i = cartesian[k][1]
        j = cartesian[k][2]
        @inbounds Er[i, j] = real(Ec[i, j])
    end
    return nothing
end


function irfft!(
    FT::FourierTransform,
    S::AbstractArray{Complex{T}, 1},
    E::AbstractArray{T, 1},
) where T
    FFTW.mul!(E, FT.pirfft, S)   # frequency -> time
    return nothing
end


function irfft2!(
    FT::FourierTransform,
    S::CuArrays.CuArray{Complex{T}, 2},
    E::CuArrays.CuArray{T, 2},
) where T
    FFTW.mul!(E, FT.pirfft2, S)   # frequency -> time
    return nothing
end


function hilbert!(
    FT::FourierTransform,
    Sr::AbstractArray{Complex{T}, 1},
    Ec::AbstractArray{Complex{T}, 1},
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
    ifft!(FT, FT.Sc, Ec)   # frequency -> time
    return nothing
end


"""
Transforms the spectruum of a real signal into the complex analytic signal.

WARNING: Needs test for odd N and low frequencies.
"""
function hilbert!(
    FT::FourierTransform,
    Sr::CuArrays.CuArray{Complex{T}, 1},
    Ec::CuArrays.CuArray{Complex{T}, 1},
) where T
    nth = FT.nthreadsNt
    nbl = FT.nblocksNt
    @CUDAnative.cuda blocks=nbl threads=nth hilbert_kernel(Sr, FT.Sc)
    ifft!(FT, FT.Sc, Ec)   # frequency -> time
    return nothing
end


function hilbert_kernel(Sr, Sc)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    Nt = length(Sc)
    Nw = length(Sr)
    for i=id:stride:Nt
        if i <= Nw
            if i == 1
                @inbounds Sc[i] = Sr[i]
            else
                @inbounds Sc[i] = FloatGPU(2.) * Sr[i]
            end
        else
            @inbounds Sc[i] = FloatGPU(0.)
        end
    end
    return nothing
end


"""
Transforms the spectruum of a real signal into the complex analytic signal.

WARNING: Needs test for odd N and low frequencies.
"""
function hilbert2!(
    FT::FourierTransform,
    Sr::CuArrays.CuArray{Complex{T}, 2},
    Ec::CuArrays.CuArray{Complex{T}, 2},
) where T
    nth = FT.nthreadsNrNt
    nbl = FT.nblocksNrNt
    @CUDAnative.cuda blocks=nbl threads=nth hilbert2_kernel(Sr, FT.Sc2)
    ifft2!(FT, FT.Sc2, Ec)   # frequency -> time
    return nothing
end


function hilbert2_kernel(Sr, Sc)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    Nr, Nt = size(Sc)
    Nr, Nw = size(Sr)
    cartesian = CartesianIndices((Nr, Nt))
    for k=id:stride:Nr*Nt
        i = cartesian[k][1]
        j = cartesian[k][2]
        if j <= Nw
            if j == 1
                @inbounds Sc[i, j] = Sr[i, j]
            else
                @inbounds Sc[i, j] = FloatGPU(2.) * Sr[i, j]
            end
        else
            @inbounds Sc[i, j] = FloatGPU(0.)
        end
    end
    return nothing
end


function convolution!(
    FT::FourierTransform,
    Hw::AbstractArray{Complex{T}, 1},
    x::AbstractArray{T, 1},
) where T
    rfft!(FT, x, FT.Sr)
    @. FT.Sr = Hw * FT.Sr
    irfft!(FT, FT.Sr, x)
    return nothing
end


function convolution!(
    FT::FourierTransform,
    Hw::CuArrays.CuArray{Complex{T}, 1},
    x::CuArrays.CuArray{T, 2},
) where T
    rfft2!(FT, x, FT.Sr2)
    nth = FT.nthreadsNrNw
    nbl = FT.nblocksNrNw
    @CUDAnative.cuda blocks=nbl threads=nth convolution2_kernel(FT.Sr2, Hw)
    irfft2!(FT, FT.Sr2, x)
    return nothing
end


function convolution2_kernel(Sr, Hw)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    Nr, Nw = size(Sr)
    cartesian = CartesianIndices((Nr, Nw))
    for k=id:stride:Nr*Nw
        i = cartesian[k][1]
        j = cartesian[k][2]
        @inbounds Sr[i, j] = Hw[j] * Sr[i, j]
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


end

module FourierGPU

import FFTW
import LinearAlgebra
import CUDAnative
import CuArrays
import CUDAdrv

const FloatGPU = Float32
const ComplexGPU = ComplexF32


struct FourierTransform
    Nr :: Int64
    Nt :: Int64
    Nw :: Int64

    Er2 :: CuArrays.CuArray{FloatGPU, 2}
    Sc :: CuArrays.CuArray{ComplexGPU, 1}
    Sc2 :: CuArrays.CuArray{ComplexGPU, 2}
    Sr :: CuArrays.CuArray{ComplexGPU, 1}

    pifft :: FFTW.Plan
    pifft2 :: FFTW.Plan
    prfft :: FFTW.Plan
    prfft2 :: FFTW.Plan
    pirfft :: FFTW.Plan

    nthreads :: Int64
    nthreads2 :: Int64
    nblocks :: Int64
    nblocks2x :: Int64
    nblocks2y :: Int64
end


function FourierTransform(Nr::Int64, Nt::Int64)
    CuArrays.allowscalar(false)   # disable slow fallback methods

    if iseven(Nt)
        Nw = div(Nt, 2) + 1
    else
        Nw = div(Nt + 1, 2)
    end

    Er2 = CuArrays.cuzeros(FloatGPU, (Nr, Nt))
    Sc = CuArrays.cuzeros(ComplexGPU, Nt)
    Sc2 = CuArrays.cuzeros(ComplexGPU, (Nr, Nt))
    Sr = CuArrays.cuzeros(ComplexGPU, Nw)

    pifft = FFTW.plan_ifft(CuArrays.cuzeros(ComplexGPU, Nt))
    pifft2 = FFTW.plan_ifft(CuArrays.cuzeros(ComplexGPU, (Nr, Nt)), [2])
    prfft = FFTW.plan_rfft(CuArrays.cuzeros(FloatGPU, Nt))
    prfft2 = FFTW.plan_rfft(CuArrays.cuzeros(FloatGPU, (Nr, Nt)), [2])
    pirfft = FFTW.plan_irfft(CuArrays.cuzeros(ComplexGPU, Nw), Nt)

    dev = CUDAnative.CuDevice(0)
    MAX_THREADS_PER_BLOCK = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
    nthreads = min(Nt, MAX_THREADS_PER_BLOCK)
    nblocks = Int(ceil(Nt / nthreads))

    nthreads2 = min(isqrt(Nr * Nt), isqrt(MAX_THREADS_PER_BLOCK))
    nblocks2x = Int(ceil(Nr / nthreads2))
    nblocks2y = Int(ceil(Nt / nthreads2))

    return FourierTransform(Nr, Nt, Nw,
                            Er2, Sc, Sc2, Sr,
                            pifft, pifft2, prfft, prfft2, pirfft,
                            nthreads, nthreads2, nblocks, nblocks2x, nblocks2y)
end


function ifft!(FT::FourierTransform,
               S::CuArrays.CuArray{ComplexGPU, 1},
               E::CuArrays.CuArray{ComplexGPU, 1})
    LinearAlgebra.mul!(E, FT.pifft, S)   # frequency -> time
    return nothing
end


function ifft2!(FT::FourierTransform,
                S::CuArrays.CuArray{ComplexGPU, 2},
                E::CuArrays.CuArray{ComplexGPU, 2})
    LinearAlgebra.mul!(E, FT.pifft2, S)   # frequency -> time
    return nothing
end


function rfft!(FT::FourierTransform,
               E::CuArrays.CuArray{FloatGPU, 1},
               S::CuArrays.CuArray{ComplexGPU, 1})
    LinearAlgebra.mul!(S, FT.prfft, E)   # time -> frequency
    return nothing
end


function rfft2!(FT::FourierTransform,
                E::CuArrays.CuArray{ComplexGPU, 2},
                S::CuArrays.CuArray{ComplexGPU, 2})
    nth = FT.nthreads2
    nblx = FT.nblocks2x
    nbly = FT.nblocks2y
    @CUDAnative.cuda blocks=(nblx, nbly) threads=(nth, nth) rfft2_kernel(E, FT.Er2)
    LinearAlgebra.mul!(S, FT.prfft2, FT.Er2)   # time -> frequency
    return nothing
end


function rfft2_kernel(Ec, Er)
    idx = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    idy = (CUDAnative.blockIdx().y - 1) * CUDAnative.blockDim().y + CUDAnative.threadIdx().y
    Nr, Nt = size(Ec)
    if (idx <= Nr) & (idy <= Nt)
        Er[idx, idy] = real(Ec[idx, idy])
    end
    return nothing
end


function irfft!(FT::FourierTransform,
                S::CuArrays.CuArray{ComplexGPU, 1},
                E::CuArrays.CuArray{FloatGPU, 1})
    LinearAlgebra.mul!(E, FT.pirfft, S)   # frequency -> time
    return nothing
end


"""
Transforms the spectruum of a real signal into the complex analytic signal.

WARNING: Needs test for odd N and low frequencies.
"""
function hilbert!(FT::FourierTransform,
                   Sr::CuArrays.CuArray{ComplexGPU, 1},
                   Ec::CuArrays.CuArray{ComplexGPU, 1})
    nth = FT.nthreads
    nbl = FT.nblocks
    @CUDAnative.cuda blocks=nbl threads=nth hilbert_kernel(Sr, FT.Sc)
    ifft!(FT, FT.Sc, Ec)   # frequency -> time
    return nothing
end


function hilbert_kernel(Sr, Sc)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    Nt = length(Sc)
    Nw = length(Sr)
    if id <= Nt
        if id <= Nw
            if id == 1
                Sc[id] = Sr[id]
            else
                Sc[id] = FloatGPU(2.) * Sr[id]
            end
        else
            Sc[id] = FloatGPU(0.)
        end
    end
    return nothing
end


"""
Transforms the spectruum of a real signal into the complex analytic signal.

WARNING: Needs test for odd N and low frequencies.
"""
function hilbert2!(FT::FourierTransform,
                   Sr::CuArrays.CuArray{ComplexGPU, 2},
                   Ec::CuArrays.CuArray{ComplexGPU, 2})
    nth = FT.nthreads2
    nblx = FT.nblocks2x
    nbly = FT.nblocks2y
    @CUDAnative.cuda blocks=(nblx, nbly) threads=(nth, nth) hilbert2_kernel(Sr, FT.Sc2)
    ifft2!(FT, FT.Sc2, Ec)   # frequency -> time
    return nothing
end


function hilbert2_kernel(Sr, Sc)
    idx = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    idy = (CUDAnative.blockIdx().y - 1) * CUDAnative.blockDim().y + CUDAnative.threadIdx().y
    Nr, Nt = size(Sc)
    Nr, Nw = size(Sr)
    if (idx <= Nr) & (idy <= Nt)
        if idy <= Nw
            if idy == 1
                Sc[idx, idy] = Sr[idx, idy]
            else
                Sc[idx, idy] = FloatGPU(2.) * Sr[idx, idy]
            end
        else
            Sc[idx, idy] = FloatGPU(0.)
        end
    end
    return nothing
end


function convolution!(FT::FourierTransform,
                      Hw::CuArrays.CuArray{ComplexGPU, 1},
                      x::CuArrays.CuArray{FloatGPU, 1})
    rfft!(FT, x, FT.Sr)
    nth = FT.nthreads
    nbl = FT.nblocks
    @CUDAnative.cuda blocks=nbl threads=nth convolution_kernel(FT.Sr, Hw)
    irfft!(FT, FT.Sr, x)
    return nothing
end


function convolution_kernel(Sr, Hw)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    Nw = length(Sr)
    if id <= Nw
        Sr[id] = Hw[id] * Sr[id]
    end
    return nothing
end


end

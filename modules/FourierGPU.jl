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
    Sr2 :: CuArrays.CuArray{ComplexGPU, 2}

    pifft :: FFTW.Plan
    pifft2 :: FFTW.Plan
    prfft :: FFTW.Plan
    prfft2 :: FFTW.Plan
    pirfft :: FFTW.Plan
    pirfft2 :: FFTW.Plan

    nthreadsNt :: Int64
    nthreadsNw :: Int64
    nblocksNt :: Int64
    nblocksNw :: Int64
    nthreadsNrNt :: Int64
    nthreadsNrNw :: Int64
    nblocksNrNt :: Int64
    nblocksNrNw :: Int64
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
    Sr2 = CuArrays.cuzeros(ComplexGPU, (Nr, Nw))

    pifft = FFTW.plan_ifft(CuArrays.cuzeros(ComplexGPU, Nt))
    pifft2 = FFTW.plan_ifft(CuArrays.cuzeros(ComplexGPU, (Nr, Nt)), [2])
    prfft = FFTW.plan_rfft(CuArrays.cuzeros(FloatGPU, Nt))
    prfft2 = FFTW.plan_rfft(CuArrays.cuzeros(FloatGPU, (Nr, Nt)), [2])
    pirfft = FFTW.plan_irfft(CuArrays.cuzeros(ComplexGPU, Nw), Nt)
    pirfft2 = FFTW.plan_irfft(CuArrays.cuzeros(ComplexGPU, (Nr, Nw)), Nt, [2])

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

    return FourierTransform(Nr, Nt, Nw,
                            Er2, Sc, Sc2, Sr, Sr2,
                            pifft, pifft2, prfft, prfft2, pirfft, pirfft2,
                            nthreadsNt, nthreadsNw, nblocksNt, nblocksNw,
                            nthreadsNrNt, nthreadsNrNw, nblocksNrNt, nblocksNrNw)
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
                E::CuArrays.CuArray{FloatGPU, 2},
                S::CuArrays.CuArray{ComplexGPU, 2})
    LinearAlgebra.mul!(S, FT.prfft2, E)   # time -> frequency
    return nothing
end


function rfft2!(FT::FourierTransform,
                E::CuArrays.CuArray{ComplexGPU, 2},
                S::CuArrays.CuArray{ComplexGPU, 2})
    nth = FT.nthreadsNrNt
    nbl = FT.nblocksNrNt
    @CUDAnative.cuda blocks=nbl threads=nth rfft2_kernel(E, FT.Er2)
    LinearAlgebra.mul!(S, FT.prfft2, FT.Er2)   # time -> frequency
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


function irfft!(FT::FourierTransform,
                S::CuArrays.CuArray{ComplexGPU, 1},
                E::CuArrays.CuArray{FloatGPU, 1})
    LinearAlgebra.mul!(E, FT.pirfft, S)   # frequency -> time
    return nothing
end


function irfft2!(FT::FourierTransform,
                 S::CuArrays.CuArray{ComplexGPU, 2},
                 E::CuArrays.CuArray{FloatGPU, 2})
    LinearAlgebra.mul!(E, FT.pirfft2, S)   # frequency -> time
    return nothing
end


"""
Transforms the spectruum of a real signal into the complex analytic signal.

WARNING: Needs test for odd N and low frequencies.
"""
function hilbert!(FT::FourierTransform,
                   Sr::CuArrays.CuArray{ComplexGPU, 1},
                   Ec::CuArrays.CuArray{ComplexGPU, 1})
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
function hilbert2!(FT::FourierTransform,
                   Sr::CuArrays.CuArray{ComplexGPU, 2},
                   Ec::CuArrays.CuArray{ComplexGPU, 2})
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


function convolution!(FT::FourierTransform,
                      Hw::CuArrays.CuArray{ComplexGPU, 1},
                      x::CuArrays.CuArray{FloatGPU, 1})
    rfft!(FT, x, FT.Sr)
    nth = FT.nthreadsNw
    nbl = FT.nblocksNw
    @CUDAnative.cuda blocks=nbl threads=nth convolution_kernel(FT.Sr, Hw)
    irfft!(FT, FT.Sr, x)
    return nothing
end


function convolution_kernel(Sr, Hw)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    Nw = length(Sr)
    for i=id:stride:Nw
        @inbounds Sr[i] = Hw[i] * Sr[i]
    end
    return nothing
end


function convolution2!(FT::FourierTransform,
                      Hw::CuArrays.CuArray{ComplexGPU, 1},
                      x::CuArrays.CuArray{FloatGPU, 2})
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


end

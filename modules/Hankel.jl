"""
Implementation of the pth-order disrete Hankel transform.

Method:
    M. Guizar-Sicairos and J.C. Gutierrez-Vega, JOSA A, 21, 53 (2004).
    http://www.opticsinfobase.org/josaa/abstract.cfm?uri=JOSAA-21-1-53

Additional info:
    http://mathematica.stackexchange.com/questions/26233/computation-of-hankel-transform-using-fft-fourier
"""
module Hankel

import SpecialFunctions
import CUDAnative
import CuArrays
import CUDAdrv

using PyCall
# @pyimport scipy.special as spec

const spec = PyCall.PyNULL()

const FloatGPU = Float32
const ComplexGPU = ComplexF32


struct HankelTransform
    R :: Float64
    Nr :: Int64
    Nt :: Int64
    r :: Array{Float64, 1}
    v :: Array{Float64, 1}
    T :: CuArrays.CuArray{ComplexGPU, 2}
    RJ :: CuArrays.CuArray{FloatGPU, 1}
    JV :: CuArrays.CuArray{FloatGPU, 1}
    VJ :: CuArrays.CuArray{FloatGPU, 1}
    JR :: CuArrays.CuArray{FloatGPU, 1}
    DM1 :: CuArrays.CuArray{ComplexGPU, 1}
    DM2 :: CuArrays.CuArray{ComplexGPU, 2}
    nthreads :: Int64
    nblocks :: Int64
    nthreads2 :: Int64
    nblocks2 :: Int64
end


function HankelTransform(R::Float64, Nr::Int64, Nt::Int64, p::Int64=0)
    copy!(spec, PyCall.pyimport_conda("scipy.special", "scipy"))
    jn_zeros = spec[:jn_zeros](p, Nr + 1)
    # jn_zeros = pycall(spec.jn_zeros, Array{Float64, 1}, p, Nr + 1)
    a = jn_zeros[1:end-1]
    aNp1 = jn_zeros[end]

    V = aNp1 / (2. * pi * R)
    J = @. abs(SpecialFunctions.besselj(p + 1, a)) / R

    r = @. a / (2. * pi * V)   # radial coordinate
    v = @. a / (2. * pi * R)   # radial frequency

    S = 2. * pi * R * V

    T = zeros((Nr, Nr))
    for i=1:Nr
        for j=1:Nr
            T[i, j] = 2. * SpecialFunctions.besselj(p, a[i] * a[j] / S) /
                      abs(SpecialFunctions.besselj(p + 1, a[i])) /
                      abs(SpecialFunctions.besselj(p + 1, a[j])) / S
        end
    end

    RJ = @. R / J
    JV = @. J / V
    VJ = @. V / J
    JR = @. J / R

    # GPU:
    CuArrays.allowscalar(false)   # disable slow fallback methods

    T = CuArrays.cu(convert(Array{ComplexGPU, 2}, T))
    RJ = CuArrays.cu(convert(Array{FloatGPU, 1}, RJ))
    JV = CuArrays.cu(convert(Array{FloatGPU, 1}, JV))
    VJ = CuArrays.cu(convert(Array{FloatGPU, 1}, VJ))
    JR = CuArrays.cu(convert(Array{FloatGPU, 1}, JR))
    DM1 = CuArrays.cuzeros(ComplexGPU, Nr)
    DM2 = CuArrays.cuzeros(ComplexGPU, (Nr, Nt))

    dev = CUDAnative.CuDevice(0)
    MAX_THREADS_PER_BLOCK = CUDAdrv.attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
    nthreads = min(Nr, MAX_THREADS_PER_BLOCK)
    nblocks = Int(ceil(Nr / nthreads))
    nthreads2 = min(Nr * Nt, MAX_THREADS_PER_BLOCK)
    nblocks2 = Int(ceil(Nr * Nt / nthreads2))

    return HankelTransform(R, Nr, Nt, r, v, T, RJ, JV, VJ, JR, DM1, DM2,
                           nthreads, nblocks, nthreads2, nblocks2)
end


function kernel1D_1(RJorVJ, f)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    N = length(f)
    for i=id:stride:N
        @inbounds f[i] = f[i] * RJorVJ[i]
    end
    return nothing
end


function kernel2D_1(RJorVJ, f)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    N1, N2 = size(f)
    cartesian = CartesianIndices((N1, N2))
    for k=id:stride:N1*N2
        i = cartesian[k][1]
        j = cartesian[k][2]
        @inbounds f[i, j] = f[i, j] * RJorVJ[i]
    end
    return nothing
end


function kernel1D_2(DM, T, f)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    N = length(f)
    for i=id:stride:N
        @inbounds DM[i] = ComplexGPU(0)
        for k=1:N
            @inbounds DM[i] = DM[i] + T[i, k] * f[k]
        end
    end
    return nothing
end


function kernel2D_2(DM, T, f)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    N1, N2 = size(f)
    cartesian = CartesianIndices((N1, N2))
    for k=id:stride:N1*N2
        i = cartesian[k][1]
        j = cartesian[k][2]
        @inbounds DM[i, j] = ComplexGPU(0)
        for m=1:N1
            @inbounds DM[i, j] = DM[i, j] + T[i, m] * f[m, j]
        end
    end
    return nothing
end


function kernel1D_3(DM, JVorJR, f)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    N = length(f)
    for i=id:stride:N
        @inbounds f[i] = DM[i] * JVorJR[i]
    end
    return nothing
end


function kernel2D_3(DM, JVorJR, f)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    N1, N2 = size(f)
    cartesian = CartesianIndices((N1, N2))
    for k=id:stride:N1*N2
        i = cartesian[k][1]
        j = cartesian[k][2]
        @inbounds f[i, j] = DM[i, j] * JVorJR[i]
    end
    return nothing
end


function dht!(ht::HankelTransform, f::CuArrays.CuArray{ComplexGPU, 1})
    nth = ht.nthreads
    nbl = ht.nblocks
    @CUDAnative.cuda blocks=nbl threads=nth kernel1D_1(ht.RJ, f)
    @CUDAnative.cuda blocks=nbl threads=nth kernel1D_2(ht.DM1, ht.T, f)
    @CUDAnative.cuda blocks=nbl threads=nth kernel1D_3(ht.DM1, ht.JV, f)
    return nothing
end


function dht(ht::HankelTransform, f::CuArrays.CuArray{ComplexGPU, 1})
    fout = copy(f)
    dht!(ht, fout)
    return fout
end


function dht!(ht::HankelTransform, f::CuArrays.CuArray{ComplexGPU, 2})
    nth = ht.nthreads2
    nbl = ht.nblocks2
    @CUDAnative.cuda blocks=nbl threads=nth kernel2D_1(ht.RJ, f)
    @CUDAnative.cuda blocks=nbl threads=nth kernel2D_2(ht.DM2, ht.T, f)
    @CUDAnative.cuda blocks=nbl threads=nth kernel2D_3(ht.DM2, ht.JV, f)
    return nothing
end


function dht(ht::HankelTransform, f::CuArrays.CuArray{ComplexGPU, 2})
    fout = copy(f)
    dht!(ht, fout)
    return fout
end


function idht!(ht::HankelTransform, f::CuArrays.CuArray{ComplexGPU, 1})
    nth = ht.nthreads
    nbl = ht.nblocks
    @CUDAnative.cuda blocks=nbl threads=nth kernel1D_1(ht.VJ, f)
    @CUDAnative.cuda blocks=nbl threads=nth kernel1D_2(ht.DM1, ht.T, f)
    @CUDAnative.cuda blocks=nbl threads=nth kernel1D_3(ht.DM1, ht.JR, f)
    return nothing
end


function idht(ht::HankelTransform, f::CuArrays.CuArray{ComplexGPU, 1})
    fout = copy(f)
    idht!(ht, fout)
    return fout
end


function idht!(ht::HankelTransform, f::CuArrays.CuArray{ComplexGPU, 2})
    nth = ht.nthreads2
    nbl = ht.nblocks2
    @CUDAnative.cuda blocks=nbl threads=nth kernel2D_1(ht.VJ, f)
    @CUDAnative.cuda blocks=nbl threads=nth kernel2D_2(ht.DM2, ht.T, f)
    @CUDAnative.cuda blocks=nbl threads=nth kernel2D_3(ht.DM2, ht.JR, f)
    return nothing
end


function idht(ht::HankelTransform, f::CuArrays.CuArray{ComplexGPU, 2})
    fout = copy(f)
    idht!(ht, fout)
    return fout
end


end

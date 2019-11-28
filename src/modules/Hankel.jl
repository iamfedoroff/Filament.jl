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

import PyCall

const FloatGPU = Float32
const MAX_THREADS_PER_BLOCK =
        CUDAdrv.attribute(
            CUDAnative.CuDevice(0),
            CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        )


struct HankelTransform{T}
    R :: Float64
    Nr :: Int
    r :: Array{Float64, 1}
    v :: Array{Float64, 1}
    TT :: CuArrays.CuArray{T, 2}
    RJ :: CuArrays.CuArray{T, 1}
    JV :: CuArrays.CuArray{T, 1}
    VJ :: CuArrays.CuArray{T, 1}
    JR :: CuArrays.CuArray{T, 1}
    DM :: CuArrays.CuArray{Complex{T}}
    nthreads :: Int
    nblocks :: Int
end


function HankelTransform(R::Float64, ndims::Int...; p::Int=0)
    Nr = ndims[1]

    spec = PyCall.pyimport("scipy.special")
    jn_zeros = spec.jn_zeros(p, Nr + 1)
    a = jn_zeros[1:end-1]
    aNp1 = jn_zeros[end]

    V = aNp1 / (2. * pi * R)
    J = @. abs(SpecialFunctions.besselj(p + 1, a)) / R

    r = @. a / (2. * pi * V)   # radial coordinate
    v = @. a / (2. * pi * R)   # radial frequency

    S = 2. * pi * R * V

    TT = zeros((Nr, Nr))
    for i=1:Nr
        for j=1:Nr
            TT[i, j] = 2. * SpecialFunctions.besselj(p, a[i] * a[j] / S) /
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

    TT = CuArrays.CuArray(convert(Array{FloatGPU, 2}, TT))
    RJ = CuArrays.CuArray(convert(Array{FloatGPU, 1}, RJ))
    JV = CuArrays.CuArray(convert(Array{FloatGPU, 1}, JV))
    VJ = CuArrays.CuArray(convert(Array{FloatGPU, 1}, VJ))
    JR = CuArrays.CuArray(convert(Array{FloatGPU, 1}, JR))
    DM = CuArrays.zeros(Complex{FloatGPU}, ndims)

    NN = length(DM)
    nthreads = min(NN, MAX_THREADS_PER_BLOCK)
    nblocks = Int(ceil(NN / nthreads))

    return HankelTransform(R, Nr, r, v, TT, RJ, JV, VJ, JR, DM, nthreads, nblocks)
end


function dht!(ht::HankelTransform, f::CuArrays.CuArray{T}) where T
    nth = ht.nthreads
    nbl = ht.nblocks
    @CUDAnative.cuda blocks=nbl threads=nth kernel1(ht.RJ, f)
    @CUDAnative.cuda blocks=nbl threads=nth kernel2(ht.DM, ht.TT, f)
    @CUDAnative.cuda blocks=nbl threads=nth kernel3(ht.DM, ht.JV, f)
    return nothing
end


function idht!(ht::HankelTransform, f::CuArrays.CuArray{T}) where T
    nth = ht.nthreads
    nbl = ht.nblocks
    @CUDAnative.cuda blocks=nbl threads=nth kernel1(ht.VJ, f)
    @CUDAnative.cuda blocks=nbl threads=nth kernel2(ht.DM, ht.TT, f)
    @CUDAnative.cuda blocks=nbl threads=nth kernel3(ht.DM, ht.JR, f)
    return nothing
end


function kernel1(RJorVJ, f::CUDAnative.CuDeviceArray{T, 1}) where T
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    N = length(f)
    for i=id:stride:N
        @inbounds f[i] = f[i] * RJorVJ[i]
    end
    return nothing
end


function kernel1(RJorVJ, f::CUDAnative.CuDeviceArray{T, 2}) where T
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


function kernel2(DM, TT, f::CUDAnative.CuDeviceArray{T, 1}) where T
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    N = length(f)
    for i=id:stride:N
        @inbounds DM[i] = 0
        for k=1:N
            @inbounds DM[i] = DM[i] + TT[i, k] * f[k]
        end
    end
    return nothing
end


function kernel2(DM, TT, f::CUDAnative.CuDeviceArray{T, 2}) where T
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    N1, N2 = size(f)
    cartesian = CartesianIndices((N1, N2))
    for k=id:stride:N1*N2
        i = cartesian[k][1]
        j = cartesian[k][2]
        @inbounds DM[i, j] = 0
        for m=1:N1
            @inbounds DM[i, j] = DM[i, j] + TT[i, m] * f[m, j]
        end
    end
    return nothing
end


function kernel3(DM, JVorJR, f::CUDAnative.CuDeviceArray{T, 1}) where T
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x + CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    N = length(f)
    for i=id:stride:N
        @inbounds f[i] = DM[i] * JVorJR[i]
    end
    return nothing
end


function kernel3(DM, JVorJR, f::CUDAnative.CuDeviceArray{T, 2}) where T
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


end

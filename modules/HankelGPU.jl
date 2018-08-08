"""
Implementation of the pth-order disrete Hankel transform.

Method:
    M. Guizar-Sicairos and J.C. Gutierrez-Vega, JOSA A, 21, 53 (2004).
    http://www.opticsinfobase.org/josaa/abstract.cfm?uri=JOSAA-21-1-53

Additional info:
    http://mathematica.stackexchange.com/questions/26233/computation-of-hankel-transform-using-fft-fourier
"""
module HankelGPU

import SpecialFunctions
import CuArrays
using PyCall
@pyimport scipy.special as spec

const FloatGPU = Float32
const ComplexGPU = Complex64

struct HankelTransform
    R :: Float64
    Nr :: Int64
    r :: Array{Float64, 1}
    v :: Array{Float64, 1}
    T_gpu :: CuArrays.CuArray{ComplexGPU, 2}
    RdivJ_gpu :: CuArrays.CuArray{FloatGPU, 1}
    JdivV_gpu :: CuArrays.CuArray{FloatGPU, 1}
    VdivJ_gpu :: CuArrays.CuArray{FloatGPU, 1}
    JdivR_gpu :: CuArrays.CuArray{FloatGPU, 1}
    F1_gpu :: CuArrays.CuArray{ComplexGPU, 1}
    F2_gpu :: CuArrays.CuArray{ComplexGPU, 1}
end


function HankelTransform(R::Float64, Nr::Int64, p::Int64=0)
    jn_zeros = pycall(spec.jn_zeros, Array{Float64, 1}, p, Nr + 1)
    a = jn_zeros[1:end-1]
    aNp1 = jn_zeros[end]

    V = aNp1 / (2. * pi * R)
    J = @. abs(SpecialFunctions.besselj(p + 1, a)) / R

    r = @. a / (2. * pi * V)   # radial coordinate
    v = @. a / (2. * pi * R)   # radial frequency

    S = 2. * pi * R * V

    T = zeros(Float64, (Nr, Nr))
    for i=1:Nr
        for j=1:Nr
            T[i, j] = 2. * SpecialFunctions.besselj(p, a[i] * a[j] / S) /
                      abs(SpecialFunctions.besselj(p + 1, a[i])) /
                      abs(SpecialFunctions.besselj(p + 1, a[j])) / S
        end
    end

    RdivJ = @. R / J
    JdivV = @. J / V
    VdivJ = @. V / J
    JdivR = @. J / R

    # GPU:
    CuArrays.allowscalar(false)   # disable slow fallback methods

    T_gpu = CuArrays.cu(convert(Array{ComplexGPU, 2}, T))
    RdivJ_gpu = CuArrays.cu(convert(Array{FloatGPU, 1}, RdivJ))
    JdivV_gpu = CuArrays.cu(convert(Array{FloatGPU, 1}, JdivV))
    VdivJ_gpu = CuArrays.cu(convert(Array{FloatGPU, 1}, VdivJ))
    JdivR_gpu = CuArrays.cu(convert(Array{FloatGPU, 1}, JdivR))
    F1_gpu = CuArrays.cuzeros(ComplexGPU, Nr)
    F2_gpu = CuArrays.cuzeros(ComplexGPU, Nr)

    # For some reason, the following line speeds up all further calculations.
    # Probably it triggers some GPU initialization.
    @. F1_gpu = F2_gpu * RdivJ_gpu

    return HankelTransform(R, Nr, r, v, T_gpu, RdivJ_gpu, JdivV_gpu, VdivJ_gpu,
                           JdivR_gpu, F1_gpu, F2_gpu)
end


function dht!(ht::HankelTransform, f_gpu::CuArrays.CuArray{ComplexGPU, 2})
    N1, N2 = size(f_gpu)
    @inbounds for j=1:N2
        f1d_gpu = f_gpu[:, j]   # this gives less allocations than preallocating f1d_gpu
        dht!(ht, f1d_gpu)
        f_gpu[:, j] = f1d_gpu
    end
    return nothing
end


function dht!(ht::HankelTransform, f_gpu::CuArrays.CuArray{ComplexGPU, 1})
    @inbounds @. ht.F1_gpu = f_gpu * ht.RdivJ_gpu
    A_mul_B!(ht.F2_gpu, ht.T_gpu, ht.F1_gpu)
    @inbounds @. f_gpu = ht.F2_gpu * ht.JdivV_gpu
    return nothing
end


function dht(ht::HankelTransform, f1_gpu::CuArrays.CuArray{ComplexGPU, 1})
    f2_gpu = copy(f1_gpu)
    dht!(ht, f2_gpu)
    return f2_gpu
end


function idht!(ht::HankelTransform, f_gpu::CuArrays.CuArray{ComplexGPU, 2})
    N1, N2 = size(f_gpu)
    @inbounds for j=1:N2
        f1d_gpu = f_gpu[:, j]   # this gives less allocations than preallocating f1d_gpu
        idht!(ht, f1d_gpu)
        f_gpu[:, j] = f1d_gpu
    end
    return nothing
end


function idht!(ht::HankelTransform, f_gpu::CuArrays.CuArray{ComplexGPU, 1})
    @inbounds @. ht.F2_gpu = f_gpu * ht.VdivJ_gpu
    A_mul_B!(ht.F1_gpu, ht.T_gpu, ht.F2_gpu)
    @inbounds @. f_gpu = ht.F1_gpu * ht.JdivR_gpu
    return nothing
end


function idht(ht::HankelTransform, f2_gpu::CuArrays.CuArray{ComplexGPU, 1})
    f1_gpu = copy(f2_gpu)
    idht!(ht, f1_gpu)
    return f1_gpu
end


end

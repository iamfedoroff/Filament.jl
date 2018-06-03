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
using PyCall
@pyimport scipy.special as spec

struct HankelTransform
    R :: Float64
    Nr :: Int64
    r :: Array{Float64, 1}
    v :: Array{Float64, 1}
    T :: Array{Float64, 2}
    RdivJ :: Array{Float64, 1}
    JdivV :: Array{Float64, 1}
    VdivJ :: Array{Float64, 1}
    JdivR :: Array{Float64, 1}
    F1 :: Array{Complex128, 1}
    F2 :: Array{Complex128, 1}
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

    F1 = zeros(Complex128, length(J))
    F2 = zeros(Complex128, length(J))

    return HankelTransform(R, Nr, r, v, T, RdivJ, JdivV, VdivJ, JdivR, F1, F2)
end


function dht!(ht::HankelTransform, f::Array{Complex128, 1})
    @inbounds @. ht.F1 = f * ht.RdivJ
    A_mul_B!(ht.F2, ht.T, ht.F1)
    @inbounds @. f = ht.F2 * ht.JdivV
    return nothing
end


function dht(ht::HankelTransform, f1::Array{Complex128, 1})
    f2 = copy(f1)
    dht!(ht, f2)
    return f2
end


function idht!(ht::HankelTransform, f::Array{Complex128, 1})
    @inbounds @. ht.F2 = f * ht.VdivJ
    A_mul_B!(ht.F1, ht.T, ht.F2)
    @inbounds @. f = ht.F1 * ht.JdivR
    return nothing
end


function idht(ht::HankelTransform, f2::Array{Complex128, 1})
    f1 = copy(f2)
    idht!(ht, f1)
    return f1
end


end

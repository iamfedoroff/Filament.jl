module TabulatedFunctions

import DelimitedFiles
import StaticArrays

# CPU / GPU specific functions -------------------------------------------------
import CUDA

mylog10(x::Float32) = CUDA.log10(x)
mylog10(x::Float64) = log10(x)

mypow10(x::Float32) = CUDA.pow(10f0, x)
mypow10(x::Float64) = 10^x
# ------------------------------------------------------------------------------


struct TFunction{
    T<:AbstractFloat,
    R<:AbstractRange{T},
    A<:Union{AbstractArray{T}, AbstractArray{Complex{T}}},
} <: Function
    x :: R
    y :: A
end


function TFunction(T::Type, fname::String, xu::AbstractFloat, yu::AbstractFloat)
    data = transpose(DelimitedFiles.readdlm(fname))
    x = data[1, :]
    y = data[2, :]

    x = x * xu
    y = y * yu

    # Most of the ionization rates are well described by rate functions (at
    # least at the intensities that correspond to the multiphoton
    # ionization). Therefore in a loglog scale they become very close to a
    # straight line which is friendly for the linear interpolation.
    @. x = log10(x)
    @. y = log10(y)

    # Check for sorted and evenly spaced x values:
    allclose(x) = all(y -> isapprox(y, x[1]), x)
    @assert issorted(x)
    @assert allclose(diff(x))

    xmin, xmax = convert(T, x[1]), convert(T, x[end])
    N = length(x)
    x = range(xmin, xmax, length=N)
    y = StaticArrays.SVector{N, T}(y)
    return TFunction(x, y)
end


function (tf::TFunction{T})(x::T) where T
    if x <= 0
        y = zero(T)   # in order to avoid -Inf in log10(0)
    else
        xlog10 = mylog10(x)
        ylog10 = linterp(xlog10, tf.x, tf.y)
        y = mypow10(ylog10)
    end
    return convert(T, y)
end


"""
Calculates the interpolated derivative of tabulated function tf in point x.
"""
function dtf(tf::TFunction{T}, x::T) where T
    dx = step(tf.x)
    if x <= tf.x[1]
        y2 = derivative(tf.x, tf.y, 2)
        y1 = derivative(tf.x, tf.y, 1)
        dydx = (y2 - y1) / dx
        res = y2 + dydx * (x - tf.x[2])
    elseif x >= tf.x[end]
        N = length(tf.x)
        yN = derivative(tf.x, tf.y, N)
        yNm1 = derivative(tf.x, tf.y, N-1)
        dydx = (yN - yNm1) / dx
        res = yNm1 + dydx * (x - tf.x[end-1])
    else
        i = Int(cld(x - tf.x[1], dx))   # number of steps from x[1] to x
        yip1 = derivative(tf.x, tf.y, i+1)
        yi = derivative(tf.x, tf.y, i)
        dydx = (yip1 - yi) / dx
        res = yi + dydx * (x - tf.x[i])
    end
    return convert(T, res)
end


"""
Derivative on a grid with constatnt step in point with index i
using the finite difference coefficients for the 4th accuracy order:
https://en.wikipedia.org/wiki/Finite_difference_coefficient
"""
function derivative(
    x::AbstractRange{T},
    y::Union{AbstractArray{T}, AbstractArray{Complex{T}}},
    i::Int,
) where T
    N = length(x)
    dx = step(x)
    if i <= 2
        c = StaticArrays.SVector{5, T}(-25/12, 4, -3, 4/3, -1/4)
        dydx = (c[1] * y[i] + c[2] * y[i+1] + c[3] * y[i+2] + c[4] * y[i+3] +
                c[5] * y[i+4]) / dx
    elseif i >= N-2
        c = StaticArrays.SVector{5, T}(25/12, -4, 3, -4/3, 1/4)
        dydx = (c[1] * y[i] + c[2] * y[i-1] + c[3] * y[i-2] + c[4] * y[i-3] +
                c[5] * y[i-4]) / dx
    else
        c = StaticArrays.SVector{5, T}(1/12, -2/3, 0, 2/3, -1/12)
        dydx = (c[1] * y[i-2] + c[2] * y[i-1] + c[3] * y[i] + c[4] * y[i+1] +
                c[5] * y[i+2]) / dx
    end
    return convert(eltype(y), dydx)
end


"""
Linear interpolation on a grid with the constant step.
"""
function linterp(
    x::T,
    xx::AbstractRange{T},
    yy::Union{AbstractArray{T}, AbstractArray{Complex{T}}},
) where T
    dx = step(xx)
    if x <= xx[1]
        dydx = (yy[2] - yy[1]) / dx
        y = yy[2] + dydx * (x - xx[2])
    elseif x >= xx[end]
        dydx = (yy[end] - yy[end-1]) / dx
        y = yy[end-1] + dydx * (x - xx[end-1])
    else
        i = Int(cld(x - xx[1], dx))   # number of steps from xx[1] to x
        i = min(i, length(xx)-1)   # extra safety
        dydx = (yy[i+1] - yy[i]) / dx
        y = yy[i] + dydx * (x - xx[i])
    end
    return convert(eltype(yy), y)
end


end

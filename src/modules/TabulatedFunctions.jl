module TabulatedFunctions

import DelimitedFiles
import StaticArrays
import CUDA


struct TFunction{R<:AbstractRange, T<:AbstractArray} <: Function
    x :: R
    y :: T
end


function TFunction(fname::String, xu::F, yu::F) where F<:AbstractFloat
    return TFunction(Float32, fname, xu, yu)
end


function TFunction(T::Type, fname::String, xu::F, yu::F) where F<:AbstractFloat
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

    N = length(x)
    x = StepRangeLen{T, T, T}(range(x[1], x[end], length=N))
    y = StaticArrays.SVector{N, T}(y)
    return TFunction(x, y)
end


function (tf::TFunction)(x::T) where T<:AbstractFloat
    if x <= 0
        res = convert(T, 0)   # in order to avoid -Inf in log10(0)
    else
        if T == Float32   # FIXME Dirty hack for launching on both CPU and GPU
            xlog10 = CUDA.log10(x)
            ylog10 = tfbase(tf, xlog10)
            res = CUDA.pow(convert(T, 10), ylog10)
        else
            xlog10 = log10(x)
            ylog10 = tfbase(tf, xlog10)
            res = 10.0^ylog10
        end
    end
    return res
end


function tfbase(tf::TFunction, x::AbstractFloat)
    dx = tf.x.step
    if x <= tf.x[1]
        dydx = (tf.y[2] - tf.y[1]) / dx
        y = tf.y[2] + dydx * (x - tf.x[2])
    elseif x >= tf.x[end]
        dydx = (tf.y[end] - tf.y[end - 1]) / dx
        y = tf.y[end - 1] + dydx * (x - tf.x[end - 1])
    else
        i = findindex(tf.x, x)
        dydx = (tf.y[i + 1] - tf.y[i]) / dx
        y = tf.y[i] + dydx * (x - tf.x[i])
    end
    return y
end


"""
Calculates the interpolated derivative of tabulated function tf in point x.
"""
function dtf(tf::TabulatedFunctions.TFunction, x::AbstractFloat)
    dx = tf.x.step
    if x <= tf.x[1]
        y2 = dtfongrid(tf, 2)
        y1 = dtfongrid(tf, 1)
        dydx = (y2 - y1) / dx
        res = y2 + dydx * (x - tf.x[2])
    elseif x >= tf.x[end]
        N = length(tf.x)
        yN = dtfongrid(tf, N)
        yNm1 = dtfongrid(tf, N - 1)
        dydx = (yN - yNm1) / dx
        res = yNm1 + dydx * (x - tf.x[end - 1])
    else
        i = findindex(tf.x, x)
        yip1 = dtfongrid(tf, i + 1)
        yi = dtfongrid(tf, i)
        dydx = (yip1 - yi) / dx
        res = yi + dydx * (x - tf.x[i])
    end
    return res
end


"""
Calculates the derivative of tabulated function tf in grid point with index i
using the finite difference coefficients for the 4th accuracy order:
https://en.wikipedia.org/wiki/Finite_difference_coefficient
"""
function dtfongrid(tf::TabulatedFunctions.TFunction, i::Int)
    T = eltype(tf.x)
    N = length(tf.x)
    h = tf.x.step
    if i <= 2
        c = @. convert(T, (-25/12, 4, -3, 4/3, -1/4))
        res = (c[1] * tf.y[i] + c[2] * tf.y[i + 1] + c[3] * tf.y[i + 2] +
               c[4] * tf.y[i + 3] + c[5] * tf.y[i + 4]) / h
    elseif i >= N - 2
        c = @. convert(T, (25/12, -4, 3, -4/3, 1/4))
        res = (c[1] * tf.y[i] + c[2] * tf.y[i - 1] + c[3] * tf.y[i - 2] +
               c[4] * tf.y[i - 3] + c[5] * tf.y[i - 4]) / h
    else
        c = @. convert(T, (1/12, -2/3, 0, 2/3, -1/12))
        res = (c[1] * tf.y[i - 2] + c[2] * tf.y[i - 1] + c[4] * tf.y[i + 1] +
               c[5] * tf.y[i + 2]) / h
    end
    return res
end


function findindex(x::AbstractArray{T}, xc::T) where T<:AbstractFloat
    ldx = (xc - x[1]) / (x[2] - x[1])   # number of steps dx from x[1] to xc
    return Int(floor(ldx)) + 1
end


end

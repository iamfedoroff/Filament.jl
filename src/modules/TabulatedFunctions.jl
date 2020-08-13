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
            ylog10 = linterp(xlog10, tf.x, tf.y)
            res = CUDA.pow(convert(T, 10), ylog10)
        else
            xlog10 = log10(x)
            ylog10 = linterp(xlog10, tf.x, tf.y)
            res = 10.0^ylog10
        end
    end
    return res
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
        res = yNm1 + dydx * (x - tf.x[end-1])
    else
        i = Int(cld(x - tf.x[1], dx))   # number of steps from x[1] to x
        yip1 = dtfongrid(tf, i+1)
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
        res = (c[1] * tf.y[i] + c[2] * tf.y[i+1] + c[3] * tf.y[i+2] +
               c[4] * tf.y[i+3] + c[5] * tf.y[i+4]) / h
    elseif i >= N - 2
        c = @. convert(T, (25/12, -4, 3, -4/3, 1/4))
        res = (c[1] * tf.y[i] + c[2] * tf.y[i-1] + c[3] * tf.y[i-2] +
               c[4] * tf.y[i-3] + c[5] * tf.y[i-4]) / h
    else
        c = @. convert(T, (1/12, -2/3, 0, 2/3, -1/12))
        res = (c[1] * tf.y[i-2] + c[2] * tf.y[i-1] + c[4] * tf.y[i+1] +
               c[5] * tf.y[i+2]) / h
    end
    return res
end


"""
Linear interpolation on a grid with the constant step.
"""
function linterp(t::AbstractFloat, tt::AbstractArray, ff::AbstractArray)
    dt = tt[2] - tt[1]
    if t <= tt[1]
        dfdt = (ff[2] - ff[1]) / dt
        f = ff[2] + dfdt * (t - tt[2])
    elseif t >= tt[end]
        dfdt = (ff[end] - ff[end-1]) / dt
        f = ff[end-1] + dfdt * (t - tt[end-1])
    else
        i = Int(cld(t - tt[1], dt))   # number of steps from tt[1] to t
        i = min(i, length(tt)-1)   # extra safety
        dfdt = (ff[i+1] - ff[i]) / dt
        f = ff[i] + dfdt * (t - tt[i])
    end
    return f
end


end

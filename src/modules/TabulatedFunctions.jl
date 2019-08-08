module TabulatedFunctions

import DelimitedFiles
import StaticArrays
import CUDAnative


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
    x = range(convert(T, x[1]), convert(T, x[end]), length=N)
    y = StaticArrays.SVector{N, T}(y)
    return TFunction(x, y)
end


function (tf::TFunction)(x::T) where T<:AbstractFloat
    xc = CUDAnative.log10(x)
    if xc < tf.x[1]
        yc = convert(T, 0)
    elseif xc >= tf.x[end]
        yc = CUDAnative.pow(convert(T, 10), tf.y[end])
    else
        i = findindex(tf.x, xc)
        dy = slope(tf.x, tf.y, i)
        yc = tf.y[i] + dy * (xc - tf.x[i])
        yc = CUDAnative.pow(convert(T, 10), yc)
    end
    return yc
end


function slope(x::AbstractArray{T}, y::AbstractArray{T}, i::Int) where T<:AbstractFloat
    if i < length(x)
        dy = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    else
        dy = (y[end] - y[end - 1]) / (x[end] - x[end - 1])
    end
    return dy
end


function findindex(x::AbstractArray{T}, xc::T) where T<:AbstractFloat
    ldx = (xc - x[1]) / (x[2] - x[1])   # number of steps dx from x[1] to xc
    return Int(floor(ldx)) + 1
end


end

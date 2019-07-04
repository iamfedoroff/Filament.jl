module TabularFunctions

import DelimitedFiles
import StaticArrays
import CUDAnative

import Units


struct TabularFunction{T<:AbstractArray} <: Function
    x :: T
    y :: T
    # dy :: T
end


function TabularFunction(T::Type, unit::Units.Unit, fname::String)
    data = transpose(DelimitedFiles.readdlm(fname))
    x = data[1, :]
    y = data[2, :]

    x = x / unit.I
    y = y * unit.t

    # Most of the ionization rates are well described by rate functions (at
    # least at the intensities that correspond to the multiphoton
    # ionization). Therefore in a loglog scale they become very close to a
    # straight line which is friendly for the linear interpolation.
    @. x = log10(x)
    @. y = log10(y)

    N = length(x)
    dy = zeros(N)
    for i=1:N-1
        dy[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    end

    x = StaticArrays.SVector{N, T}(x)
    y = StaticArrays.SVector{N, T}(y)
    dy = StaticArrays.SVector{N, T}(dy)

    # return TabularFunction(x, y, dy)
    return TabularFunction(x, y)
end


function (tf::TabularFunction)(x::T) where T<:AbstractFloat
    # xc = log10(x)
    xc = CUDAnative.log10(x)
    if xc < tf.x[1]
        yc = convert(T, 0)
    elseif xc >= tf.x[end]
        # yc = convert(T, 10)^tf.y[end]
        yc = CUDAnative.pow(convert(T, 10), tf.y[end])
    else
        ic = searchsorted(tf.x, xc)
        dy = tangent(tf.x, tf.y, ic)
        yc = tf.y[ic] + dy * (xc - tf.x[ic])
        # yc = tf.y[ic] + tf.dy[ic] * (xc - tf.x[ic])
        # yc = convert(T, 10)^yc
        yc = CUDAnative.pow(convert(T, 10), yc)
    end
    return yc
end


function tangent(x::AbstractArray{T}, y::AbstractArray{T}, i::Int) where T<:AbstractFloat
    if i <= length(x) - 1
        dy = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    else
        dy = convert(T, 0)
    end
    return dy
end


function searchsorted(x::AbstractArray, xc::AbstractFloat)
    xcnorm = (xc - x[1]) / (x[end] - x[1])
    # return Int(floor(xcnorm * length(x) + 1))
    return Int(CUDAnative.floor(xcnorm * length(x) + 1))
end


end

module TabularFunctions

import DelimitedFiles
import CUDAnative
import CuArrays

import Units


struct TabularFunction{T}
    x :: AbstractArray{T, 1}
    y :: AbstractArray{T, 1}
    dy :: AbstractArray{T, 1}
end


function TabularFunction(unit::Units.Unit, fname::String)
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

    return TabularFunction(x, y, dy)
end


function CuTabularFunction(T::Type, unit::Units.Unit, fname::String)
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

    x = CuArrays.CuArray(convert(Array{T, 1}, x))
    y = CuArrays.CuArray(convert(Array{T, 1}, y))
    dy = CuArrays.CuArray(convert(Array{T, 1}, dy))

    return TabularFunction(x, y, dy)
end


function tfvalue(tf::TabularFunction, xval::T) where T<:AbstractFloat
    xc = log10(xval)
    if xc < tf.x[1]
        y = convert(T, 0)
    elseif xc >= tf.x[end]
        y = convert(T, 10)^tf.y[end]
    else
        i = searchsortedfirst(tf.x, xc)
        y = tf.y[i] + tf.dy[i] * (xc - tf.x[i])
        y = convert(T, 10)^y
    end
    return y
end


function tfvalue(xval::T, x::AbstractArray{T, 1}, y::AbstractArray{T, 1},
                 dy::AbstractArray{T, 1}) where T<:AbstractFloat
    xc = CUDAnative.log10(xval)
    if xc < x[1]
        yc = convert(T, 0)
    elseif xc >= x[end]
        yc = CUDAnative.pow(convert(T, 10), y[end])
    else
        ic = searchsorted(x, xc)
        yc = y[ic] + dy[ic] * (xc - x[ic])
        yc = CUDAnative.pow(convert(T, 10), yc)
    end
    return yc
end


function searchsorted(x::AbstractArray{T, 1}, xc::T) where T<:AbstractFloat
    xcnorm = (xc - x[1]) / (x[end] - x[1])
    return Int(CUDAnative.floor(xcnorm * length(x) + 1))
end


end

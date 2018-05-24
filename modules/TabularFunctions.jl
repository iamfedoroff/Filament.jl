module TabularFunctions

using Interpolations


struct TabularFunction
    x :: Array{Float64, 1}
    y :: Array{Float64, 1}
end


function TabularFunction(unit, fname::String)
    data = transpose(readdlm(fname))
    x = data[1, :]
    y = data[2, :]

    x = x / unit.I
    y = y * unit.t

    # Most of the ionization rates are well described by rate functions (at
    # least at the intensities that correspond to the multiphoton
    # ionization). Therefore in a loglog scale they become very close to a
    # straight line which is friendly for the linear interpolation.
    x = log10.(x)
    y = log10.(y)

    return TabularFunction(x, y)
end


function tfvalue(tf::TabularFunction, x::Float64)
    if x == 0.
        y = 0.
    else
        y = interp(log10(x), tf.x, tf.y)
        y = 10.^y
    end
    return y
end


function interp(xi::Float64, x::Array{Float64, 1}, y::Array{Float64, 1})
    # itp = interpolate(y, BSpline(Linear()), OnGrid())
    # itp = scale(itp, x)
    itp = interpolate((x,), y, Gridded(Linear()))
    itp = extrapolate(itp, Flat())
    yi = itp[xi]
    return yi
end


end

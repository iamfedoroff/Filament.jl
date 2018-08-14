module TabularFunctions


struct TabularFunction
    x :: Array{Float64, 1}
    y :: Array{Float64, 1}
    dum :: Array{Float64, 1}
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
    @. x = log10(x)
    @. y = log10(y)

    dum = zeros(length(x))

    return TabularFunction(x, y, dum)
end


function tfvalue(tf::TabularFunction, x::Float64)
    xc = log10(x)
    if xc < tf.x[1]
        y = 0.
    elseif xc >= tf.x[end]
        y = 10.^tf.y[end]
    else
        @inbounds @. tf.dum = abs(tf.x - xc)
        i = indmin(tf.dum)
        y = tf.y[i] + (tf.y[i + 1] - tf.y[i]) * (xc - tf.x[i]) /
                      (tf.x[i + 1] - tf.x[i])
        y = 10.^y
    end
    return y
end


end

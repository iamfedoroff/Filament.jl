module TabularFunctions

import DelimitedFiles


struct TabularFunction
    x :: Array{Float64, 1}
    y :: Array{Float64, 1}
    dy :: Array{Float64, 1}
end


function TabularFunction(unit, fname::String)
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


function tfvalue(tf::TabularFunction, x::Float64)
    xc = log10(x)
    if xc < tf.x[1]
        y = 0.
    elseif xc >= tf.x[end]
        y = 10. ^ tf.y[end]
    else
        i = searchsortedfirst(tf.x, xc)
        y = tf.y[i] + tf.dy[i] * (xc - tf.x[i])
        y = 10. ^ y
    end
    return y
end


end

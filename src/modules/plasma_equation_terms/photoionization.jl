function init_photoionization(unit, n0, w0, params::Dict)
    rho_nt = params["rho_nt"]
    EREAL = params["EREAL"]
    components = params["components"]

    fiarg_real(x::Complex{FloatGPU}) = real(x)^2
    fiarg_abs2(x::Complex{FloatGPU}) = abs2(x)
    if EREAL
        fiarg = fiarg_real
    else
        fiarg = fiarg_abs2
    end

    Neq = length(components)
    frho0s = zeros(Neq)
    tabfuncs = Array{Function}(undef, Neq)
    for (i, comp) in enumerate(components)
        frac = comp["fraction"]
        frho0s[i] = frac * rho_nt / unit.rho

        fname = comp["tabular_function"]
        tabfuncs[i] = TabularFunctions.TabularFunction(FloatGPU, unit, fname)
    end
    frho0s = StaticArrays.SVector{Neq, FloatGPU}(frho0s)
    tabfuncs = StaticArrays.SVector{Neq}(tabfuncs)

    R = FloatGPU(1.)
    p = (frho0s, tabfuncs, fiarg)
    pcalc = Equations.PFunction(calc_photoionization, p)
    return Equations.Term(R, pcalc, Neq)
end


function calc_photoionization(rho::StaticArrays.SVector, args, p)
    frho0s, tabfuncs, fiarg = p
    E, = args

    I = fiarg(E)

    Neq = length(rho)
    drho = StaticArrays.SVector{Neq, FloatGPU}(rho)
    for i=1:Neq
        R1 = tabfuncs[i](I)
        tmp = R1 * (frho0s[i] - rho[i])
        drho = StaticArrays.setindex(drho, tmp, i)
    end
    return drho
end

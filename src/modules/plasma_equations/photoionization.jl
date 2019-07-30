function init_photoionization(unit, n0, w0, params)
    alg = params["ALG"]
    EREAL = params["EREAL"]
    rho_nt = params["rho_nt"]
    components = params["components"]

    fiarg_real(x::Complex{FloatGPU}) = real(x)^2
    fiarg_abs2(x::Complex{FloatGPU}) = abs2(x)
    if EREAL
        fiarg = fiarg_real
    else
        fiarg = fiarg_abs2
    end

    Ncomp = length(components)
    tabfuncs = Array{Function}(undef, Ncomp)
    frhonts = zeros(Ncomp)
    Ks = zeros(Ncomp)

    for (i, comp) in enumerate(components)
        frac = comp["fraction"]
        Ui = comp["ionization_energy"]
        tfname = comp["ionization_rate"]

        # Photoionization:
        tf = TabulatedFunctions.TFunction(FloatGPU, tfname, 1/unit.I, unit.t)
        tabfuncs[i] = tf

        frhont = frac * rho_nt
        frhont = frhont / unit.rho
        frhonts[i] = frhont

        # K * drho/dt:
        Ui = Ui * QE   # eV -> J
        Ks[i] = ceil(Ui / (HBAR * w0))
    end

    tabfuncs = StaticArrays.SVector{Ncomp}(tabfuncs)
    frhonts = StaticArrays.SVector{Ncomp, FloatGPU}(frhonts)
    Ks = StaticArrays.SVector{Ncomp, FloatGPU}(Ks)

    # Problem:
    Neq = Ncomp   # number of equations
    rho0 = StaticArrays.SVector{Neq, FloatGPU}(zeros(Neq))   # initial condition
    p = (tabfuncs, fiarg, frhonts)   # step function parameters
    pstepfunc = Equations.PFunction(stepfunc_photoionization, p)
    prob = Equations.Problem(alg, rho0, pstepfunc)

    # Function to extract electron density out of the problem solution:
    extract(u::StaticArrays.SVector) = sum(u)

    # Function to calculate K * drho/dt:
    p = (tabfuncs, fiarg, frhonts, Ks)
    kdrho_func = Equations.PFunction(kdrho_photoionization, p)

    return prob, extract, kdrho_func
end


function stepfunc_photoionization(rho, args, p)
    tabfuncs, fiarg, frhonts = p
    E, = args

    I = fiarg(E)

    Neq = length(rho)
    drho = StaticArrays.SVector{Neq, FloatGPU}(rho)
    for i=1:Neq
        tf = tabfuncs[i]
        R1 = tf(I)

        frhont = frhonts[i]

        tmp = R1 * (frhont - rho[i])
        drho = StaticArrays.setindex(drho, tmp, i)
    end
    return drho
end


function kdrho_photoionization(rho, args, p)
    tabfuncs, fiarg, frhonts, Ks = p
    E, = args

    I = fiarg(E)

    Neq = length(rho)
    kdrho = convert(FloatGPU, 0)
    for i=1:Neq
        tf = tabfuncs[i]
        R1 = tf(I)

        frhont = frhonts[i]

        drho = R1 * (frhont - rho[i])
        kdrho = kdrho + Ks[i] * drho
    end
    return kdrho
end

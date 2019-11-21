function init_photoionization(unit, n0, w0, params)
    ALG = params["ALG"]
    EREAL = params["EREAL"]
    KDEP = params["KDEP"]
    rho0 = params["rho0"]
    rho_nt = params["rho_nt"]
    components = params["components"]

    # Dirty hack which allows to launch T geometry on CPU with Float64:
    TFloat = FloatGPU
    if typeof(unit) == Units.UnitT{Float64}
        TFloat = Float64
    end

    fiarg_real(x::Complex) = real(x)^2
    fiarg_abs2(x::Complex) = abs2(x)
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
        Uiev = comp["ionization_energy"]
        tfname = comp["ionization_rate"]

        # Photoionization:
        tf = TabulatedFunctions.TFunction(TFloat, tfname, 1/unit.I, unit.t)
        tabfuncs[i] = tf

        frhont = frac * rho_nt
        frhont = frhont / unit.rho
        frhonts[i] = frhont

        # K * drho/dt:
        Ui = Uiev * QE   # eV -> J
        Ks[i] = ceil(Ui / (HBAR * w0))
    end
    tabfuncs = StaticArrays.SVector{Ncomp}(tabfuncs)
    frhonts = StaticArrays.SVector{Ncomp, TFloat}(frhonts)
    Ks = StaticArrays.SVector{Ncomp, TFloat}(Ks)

    # Initial condition:
    Neq = Ncomp   # number of equations
    rho0u = rho0 / unit.rho
    rho0u = ones(Neq) * rho0u
    rho0u = StaticArrays.SVector{Neq, TFloat}(rho0u)

    # Problem:
    p = (tabfuncs, fiarg, frhonts)   # step function parameters
    pfunc = Equations.PFunction(func_photoionization, p)
    prob = Equations.Problem(ALG, rho0u, pfunc)

    # Function to extract electron density out of the problem solution:
    extract(u::StaticArrays.SVector) = sum(u)

    # Function to calculate K * drho/dt:
    p = (tabfuncs, fiarg, frhonts, Ks, KDEP)
    kdrho_func = Equations.PFunction(kdrho_photoionization, p)

    return prob, extract, kdrho_func
end


function func_photoionization(
    rho::AbstractArray{T},
    t::T,
    args::Tuple,
    p::Tuple,
) where T<:AbstractFloat
    tabfuncs, fiarg, frhonts = p
    E, = args

    I = fiarg(E)

    Neq = length(rho)
    drho = StaticArrays.SVector{Neq, T}(rho)
    for i=1:Neq
        tf = tabfuncs[i]
        R1 = tf(I)

        frhont = frhonts[i]

        tmp = R1 * (frhont - rho[i])
        drho = StaticArrays.setindex(drho, tmp, i)
    end
    return drho
end


function kdrho_photoionization(
    rho::AbstractArray{T},
    t::T,
    args::Tuple,
    p::Tuple,
) where T<:AbstractFloat
    tabfuncs, fiarg, frhonts, Ks, KDEP = p
    E, = args

    I = fiarg(E)
    if KDEP
        if I <= 0
            Ilog = convert(T, -30)   # I=1e-30 in order to avoid -Inf in log(0)
        else
            if T == Float32   # FIXME Dirty hack for launching on both CPU and GPU
                Ilog = CUDAnative.log10(I)
            else
                Ilog = log10(I)
            end
        end
    end

    Neq = length(rho)
    kdrho = convert(T, 0)
    for i=1:Neq
        tf = tabfuncs[i]
        R1 = tf(I)

        frhont = frhonts[i]

        if KDEP
            K = TabulatedFunctions.dtf(tf, Ilog)
        else
            K = Ks[i]
        end

        drho = R1 * (frhont - rho[i])
        kdrho = kdrho + K * drho
    end
    return kdrho
end

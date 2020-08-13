function init_photoionization(unit, grid, field, medium, params)
    ALG = params["ALG"]
    EREAL = params["EREAL"]
    KDEP = params["KDEP"]
    rho0 = params["rho0"]
    rho_nt = params["rho_nt"]
    components = params["components"]

    # Dirty hack which allows to launch T geometry on CPU with Float64:
    if isa(grid, Grids.GridT)
        TFloat = Float64
    else
        TFloat = FloatGPU
    end

    w0 = field.w0
    n0 = Media.refractive_index(medium, w0)

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
    if isa(grid, Grids.GridT)
        p = (tabfuncs, fiarg, frhonts, grid.t, field.E)
        prob = ODEIntegrators.Problem(func_photoionization, rho0u, p)
        integs = ODEIntegrators.Integrator(prob, ALG)
    else
        integs = Array{ODEIntegrators.Integrator}(undef, grid.Nr)
        for i=1:grid.Nr
            Ei = CUDA.cudaconvert(view(field.E, i, :))
            pi = (tabfuncs, fiarg, frhonts, grid.t, Ei)
            probi = ODEIntegrators.Problem(func_photoionization, rho0u, pi)
            integs[i] = ODEIntegrators.Integrator(probi, ALG)
        end
        integs = CUDA.CuArray(hcat([integs[i] for i in 1:grid.Nr]))
    end

    # Function to extract electron density out of the problem solution:
    extract(u::StaticArrays.SVector) = sum(u)

    # Function to calculate K * drho/dt:
    kdrho_func = kdrho_photoionization

    if isa(grid, Grids.GridT)
        kdrho_ps = (tabfuncs, fiarg, frhonts, Ks, KDEP, grid.t, field.E)
    else
        kdrho_ps = Array{Tuple}(undef, grid.Nr)
        for i=1:grid.Nr
            Ei = CUDA.cudaconvert(view(field.E, i, :))
            kdrho_ps[i] = (tabfuncs, fiarg, frhonts, Ks, KDEP, grid.t, Ei)
        end
        kdrho_ps = CUDA.CuArray(hcat([kdrho_ps[i] for i in 1:grid.Nr]))
    end

    return integs, extract, kdrho_func, kdrho_ps
end


function func_photoionization(
    rho::AbstractArray{T},
    p::Tuple,
    t::T,
) where T<:AbstractFloat
    tabfuncs, fiarg, frhonts, tt, EE = p

    E = linterp(t, tt, EE)
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
    p::Tuple,
    t::T,
) where T<:AbstractFloat
    tabfuncs, fiarg, frhonts, Ks, KDEP, tt, EE = p

    E = linterp(t, tt, EE)
    I = fiarg(E)
    if KDEP
        if I <= 0
            Ilog = convert(T, -30)   # I=1e-30 in order to avoid -Inf in log(0)
        else
            if T == Float32   # FIXME Dirty hack for launching on both CPU and GPU
                Ilog = CUDA.log10(I)
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


function linterp(t::AbstractFloat, tt::AbstractArray, ff::AbstractArray)
    if t <= tt[1]
        f = ff[1]
    elseif t >= tt[end]
        f = ff[end]
    else
        dt = tt[2] - tt[1]
        i = Int(cld(t - tt[1], dt))   # number of steps from tt[1] to t
        f = ff[i] + (ff[i+1] - ff[i]) / (tt[i+1] - tt[i]) * (t - tt[i])
    end
    return f
end

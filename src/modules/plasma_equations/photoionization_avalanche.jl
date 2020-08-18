function init_photoionization_avalanche(t, E, w0, units, params)
    ALG = params["ALG"]
    EREAL = params["EREAL"]
    KDEP = params["KDEP"]
    rho0 = params["rho0"]
    rho_nt = params["rho_nt"]
    nuc = params["nuc"]
    mr = params["mr"]
    components = params["components"]

    tu, Eu, Iu, rhou = units

    TFloat = eltype(t)   # allows to launch T geometry on CPU with Float64

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
    Ravas = zeros(Ncomp)
    Ks = zeros(Ncomp)
    for (i, comp) in enumerate(components)
        frac = comp["fraction"]
        Uiev = comp["ionization_energy"]
        tfname = comp["ionization_rate"]

        # Photoionization:
        tf = TabulatedFunctions.TFunction(TFloat, tfname, 1/Iu, tu)
        tabfuncs[i] = tf

        frhont = frac * rho_nt
        frhont = frhont / rhou
        frhonts[i] = frhont

        # Impact ionization:
        Ui = Uiev * QE   # eV -> J
        MR = mr * ME   # reduced mass of electron and hole (effective mass)
        sigmaB = QE^2 / MR * nuc / (nuc^2 + w0^2)
        Rava = sigmaB / Ui
        Rava = Rava * tu * Eu^2
        Ravas[i] = Rava

        # K * drho/dt:
        Ks[i] = ceil(Ui / (HBAR * w0))
    end
    tabfuncs = StaticArrays.SVector{Ncomp}(tabfuncs)
    frhonts = StaticArrays.SVector{Ncomp, TFloat}(frhonts)
    Ravas = StaticArrays.SVector{Ncomp, TFloat}(Ravas)
    Ks = StaticArrays.SVector{Ncomp, TFloat}(Ks)

    # Initial condition:
    rho0u = ones(Ncomp) * rho0 / rhou
    rho0u = StaticArrays.SVector{Ncomp, TFloat}(rho0u)

    # Problem:
    if ndims(E) == 1
        p = (tabfuncs, fiarg, frhonts, Ravas, t, E)
        prob = ODEIntegrators.Problem(func_photoionization_avalanche, rho0u, p)
        integs = ODEIntegrators.Integrator(prob, ALG)

        p = (tabfuncs, fiarg, frhonts, t, E)
        prob = ODEIntegrators.Problem(func_photoionization, rho0u, p)
        kdrho_integs = ODEIntegrators.Integrator(prob, ALG)
    else
        Nr, Nt = size(E)
        integs = Array{ODEIntegrators.Integrator}(undef, Nr)
        kdrho_integs = Array{ODEIntegrators.Integrator}(undef, Nr)
        for i=1:Nr
            Ei = CUDA.cudaconvert(view(E, i, :))

            pi = (tabfuncs, fiarg, frhonts, Ravas, t, Ei)
            probi = ODEIntegrators.Problem(func_photoionization_avalanche, rho0u, pi)
            integs[i] = ODEIntegrators.Integrator(probi, ALG)

            pi = (tabfuncs, fiarg, frhonts, t, Ei)
            probi = ODEIntegrators.Problem(func_photoionization, rho0u, pi)
            kdrho_integs[i] = ODEIntegrators.Integrator(probi, ALG)
        end
        integs = CUDA.CuArray(hcat([integs[i] for i in 1:Nr]))
        kdrho_integs = CUDA.CuArray(hcat([kdrho_integs[i] for i in 1:Nr]))
    end

    # Function to extract electron density out of the problem solution:
    extract(u::StaticArrays.SVector) = sum(u)

    # Function to calculate K * drho/dt:
    if ndims(E) == 1
        kdrho_ps = (tabfuncs, fiarg, frhonts, Ks, KDEP, t, E)
    else
        Nr, Nt = size(E)
        kdrho_ps = Array{Tuple}(undef, Nr)
        for i=1:Nr
            Ei = CUDA.cudaconvert(view(E, i, :))
            kdrho_ps[i] = (tabfuncs, fiarg, frhonts, Ks, KDEP, t, Ei)
        end
        kdrho_ps = CUDA.CuArray(hcat([kdrho_ps[i] for i in 1:Nr]))
    end

    return integs, extract, kdrho_integs, kdrho_func, kdrho_ps
end


function func_photoionization_avalanche(
    rho::AbstractArray{T},
    p::Tuple,
    t::T,
) where T<:AbstractFloat
    tabfuncs, fiarg, frhonts, Ravas, tt, EE = p

    E = TabulatedFunctions.linterp(t, tt, EE)
    I = fiarg(E)
    E2 = real(E)^2

    rho_tot = sum(rho)   # total electron density

    Ncomp = length(rho)
    drho = StaticArrays.SVector{Ncomp, T}(rho)
    for i=1:Ncomp
        tf = tabfuncs[i]
        frhont = frhonts[i]
        Rava = Ravas[i]

        R1 = tf(I)
        R2 = Rava * E2

        tmp = R1 * (frhont - rho[i]) + R2 * (frhont - rho[i]) / frhont * rho_tot
        drho = StaticArrays.setindex(drho, tmp, i)
    end
    return drho
end


function func_photoionization(
    rho::AbstractArray{T},
    p::Tuple,
    t::T,
) where T<:AbstractFloat
    tabfuncs, fiarg, frhonts, tt, EE = p

    E = TabulatedFunctions.linterp(t, tt, EE)
    I = fiarg(E)

    Ncomp = length(rho)
    drho = StaticArrays.SVector{Ncomp, T}(rho)
    for i=1:Ncomp
        tf = tabfuncs[i]
        frhont = frhonts[i]

        R1 = tf(I)

        tmp = R1 * (frhont - rho[i])
        drho = StaticArrays.setindex(drho, tmp, i)
    end
    return drho
end


function kdrho_func(
    rho::AbstractArray{T},
    p::Tuple,
    t::T,
) where T<:AbstractFloat
    tabfuncs, fiarg, frhonts, Ks, KDEP, tt, EE = p

    E = TabulatedFunctions.linterp(t, tt, EE)
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

    Ncomp = length(rho)
    kdrho = convert(T, 0)
    for i=1:Ncomp
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

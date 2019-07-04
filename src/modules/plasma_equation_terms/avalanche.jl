function init_avalanche(unit, n0, w0, params::Dict)
    mr = params["mr"]
    nuc = params["nuc"]
    components = params["components"]

    Neq = length(components)
    Ravas = zeros(Neq)
    for (i, comp) in enumerate(components)
        Ui = comp["ionization_energy"]
        Ui = Ui * QE   # eV -> J
        MR = mr * ME   # reduced mass of electron and hole (effective mass)
        sigmaB = QE^2 / MR * nuc / (nuc^2 + w0^2)
        Ravas[i] = sigmaB / Ui
    end
    Eu = Units.E(unit, n0)
    Ravas = Ravas * unit.t * Eu^2
    Ravas = StaticArrays.SVector{Neq, FloatGPU}(Ravas)

    R = FloatGPU(1.)
    p = (Ravas, )
    pcalc = Equations.PFunction(calc_avalanche, p)
    return Equations.Term(R, pcalc, Neq)
end


function calc_avalanche(rho::StaticArrays.SVector, args, p)
    Ravas, = p
    E, = args

    E2 = real(E)^2

    Neq = length(rho)
    drho = StaticArrays.SVector{Neq, FloatGPU}(rho)
    for i=1:Neq
        R2 = Ravas[i] * E2
        tmp = R2 * rho[i]
        drho = StaticArrays.setindex(drho, tmp, i)
    end
    return drho
end

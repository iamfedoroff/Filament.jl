# ------------------------------------------------------------------------------
# Linear response
# ------------------------------------------------------------------------------
function permittivity(w)
    # E.R. Peck and K. Reeder "Dispersion of Air" JOSA, 62, 958 (1972)
    if w == 0.
        sig = 0.
    else
        lam = 2. * pi * C0 / w
        sig = 1. / lam
        sig = sig * 1e-6   # 1/m -> 1/um
    end
    dum = 8060.51 + 2480990. / (132.247 - sig^2) + 17455.7 / (39.32957 - sig^2)
    n = 1. + dum * 1e-8
    eps = n^2
    return eps
end


function permeability(w)
    mu = 1.
    return mu
end


# ------------------------------------------------------------------------------
# Nonlinear response
# ------------------------------------------------------------------------------
# DEFPATH - default path for directory with media responses

n2 = 1e-23   # [m**2/W] nonlinear index
graman = 0.5   # fraction of stimulated Raman contribution

rho0 = 2.5e25   # [1/m**3] neutrals density [https://en.wikipedia.org/wiki/Number_density]
nuc = 5e12   # [1/s] collision frequency [Sprangle, PRE, 69, 066415 (2004)]
mr = 1.   # [me] reduced mass of electron and hole (effective mass)


# Cubic nonlinearity -----------------------------------------------------------
include(joinpath(DEFPATH, "cubic.jl"))

cubic = Dict(
    "init" => init_cubic,   # initialization function
    "THG" => true,   # switch for third harmonic generation
    "n2" => (1. - graman) * n2,   # [m**2/W] nonlinear index
    )


# Stimulated Raman effect ------------------------------------------------------
function raman_response(t)
    # M. Mlejnek, E.M. Wright, J.V. Moloney "Dynamic spatial replenishment of
    # femtosecond pulses propagating in air" Opt. Lett., 23, 382 (1998)
    if t < 0.
        H = 0.
    else
        Omega = 20.6e12
        Gamma = 26e12
        Lambda = sqrt(Omega^2 - Gamma^2 / 4.)
        H = Omega^2 * exp(-0.5 * Gamma * t) * sin(Lambda * t) / Lambda
    end
    return H
end


include(joinpath(DEFPATH, "raman.jl"))

raman = Dict(
    "init" => init_raman,   # initialization function
    "THG" => true,   # switch for third harmonic generation
    "n2" =>  graman * n2,   # [m**2/W] nonlinear index
    "raman_response" => raman_response,   # response function
    )


# Free current -----------------------------------------------------------------
include(joinpath(DEFPATH, "current_free.jl"))

current_free = Dict(
    "init" => init_current_free,   # initialization function
    "nuc" => nuc,   # [1/s] collision frequency [Sprangle, PRE, 69, 066415 (2004)]
    "mr" => mr,   # [me] reduced mass of electron and hole (effective mass)
    )


# Multiphoton absorption -------------------------------------------------------
include(joinpath(DEFPATH, "current_losses.jl"))

current_losses = Dict(
    "init" => init_current_losses,   # initialization function
    "EREAL" => false,   # switch for the ionization rate argument: real(E)^2 vs abs2(E)
    )


# Lattice ----------------------------------------------------------------------
# The perturbation of the linear refractive index dn(r, z) is given by
#     dn(r, z) = dnr(r) * dnz(z),
# where dnr and dnz define the shape of the perturbation along r and z
# coordinates, respectively.
#
function dnr_func(r, ru)
    A = -1e-6   # amplitude of the perturbation
    ar = 100e-6   # [m] radius of the perturbation
    return A * exp(-(r * ru)^2 / ar^2)
end


function dnz_func(z, zu)
    zstart = 1.5   # [m] distance z where the perturbation starts
    zend = 2.5   # [m] distance z where the perturbation ends
    az = 0.5 * (zend - zstart)
    return exp(-((z * zu - (zstart + az)) / az)^10)
end


include("lattice.jl")

lattice = Dict(
    "init" => init_lattice,
    "dnr_func" => dnr_func,
    "dnz_func" => dnz_func,
    )


# List of nonlinear responses included in the model:
responses = [cubic, raman, current_free, current_losses, lattice]


# ------------------------------------------------------------------------------
# Equation for electron density
# ------------------------------------------------------------------------------
# Components:
# Multiphoton ionization rates are from [Kasparian, APB, 71, 877 (2000)]
N2 = Dict("name" => "nitrogen",
          "fraction" => 0.79,
          "ionization_energy" => 15.576,   # in eV
          "tabular_function" => "multiphoton_N2.tf")

O2 = Dict("name" => "oxygen",
          "fraction" => 0.21,
          "ionization_energy" => 12.063,   # in eV
          "tabular_function" => "multiphoton_O2.tf")

components = [N2, O2]


plasma_equation = Dict(
    "METHOD" => "ETD",   # numerical method (ETD, RK2, RK3, RK4)
    "AVALANCHE" => true,   # switch for avalanche ionization
    "EREAL" => false,   # switch for the ionization rate argument: real(E)^2 vs abs2(E)
    "KGAMMA" => true,   # multiphoton ionization order K depends on Keldysh gamma
    "rho0" => rho0,   # [1/m**3] neutrals density
    "nuc" => nuc,   # [1/s] collision frequency
    "mr" => mr,   # [me] reduced mass of electron and hole (effective mass)
    "components" => components,
    )

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


n2 = 1e-23   # [m**2/W] nonlinear index

rho0 = 2.5e25   # [1/m**3] neutrals density [https://en.wikipedia.org/wiki/Number_density]
nuc = 5e12   # [1/s] collision frequency [Sprangle, PRE, 69, 066415 (2004)]
mr = 1.   # [me] reduced mass of electron and hole (effective mass)

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

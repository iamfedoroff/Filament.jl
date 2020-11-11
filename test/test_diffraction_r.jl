here = pwd()

lam0 = 800e-9   # [m] central wavelength
a0 = 1e-3   # [m] initial beam radius
zd = 2 * pi / lam0 * a0^2   # [m] diffraction length


# ******************************************************************************
# R
# ******************************************************************************
# Run simulations:
fname = joinpath("diffraction", "R", "input.jl")
input = Filament.prepare(fname)
Filament.run(input)

# Compare to theory:
fname = joinpath("results", "plot.dat")
data = DelimitedFiles.readdlm(fname, comments=true)
data = transpose(data)
z = data[1, :]   # [zu] propagation distance
I = data[2, :]   # [Iu] itensity
a = data[3, :] * 1e-3   # [m] 1/e beam radius

ath = @. a0 * sqrt(1 + (z / zd)^2)   # [m] theoretical 1/e beam radius
Ith = @. (a0 / ath)^2   # [I0] theoretical intensity

@test maximum(abs.(a .- ath)) <= 0.02e-3   # difference is less than the step size
@test isapprox(I, Ith, rtol=1e-3)   # difference is less than 0.1%

# Delete output files:
rm("results", recursive=true)
rm("ht.jld2")

cd(here)


# ******************************************************************************
# Rn
# ******************************************************************************
# Run simulations:
fname = joinpath("diffraction", "Rn", "input.jl")
input = Filament.prepare(fname)
Filament.run(input)

# Compare to theory:
fname = joinpath("results", "plot.dat")
data = DelimitedFiles.readdlm(fname, comments=true)
data = transpose(data)
z = data[1, :]   # [zu] propagation distance
I = data[2, :]   # [Iu] itensity
a = data[3, :] * 1e-3   # [m] 1/e beam radius

ath = @. a0 * sqrt(1 + (z / zd)^2)   # [m] theoretical 1/e beam radius
Ith = @. (a0 / ath)^2   # [I0] theoretical intensity

@test maximum(abs.(a .- ath)) <= 0.02e-3   # difference is less than the step size
@test isapprox(I, Ith, rtol=1e-3)   # difference is less than 0.1%

# Delete output files:
rm("results", recursive=true)

cd(here)

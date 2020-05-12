here = pwd()

# Run simulations:
fname = joinpath("dispersion", "T", "input.jl")
input = Filament.prepare(fname)
Filament.run(input)

# Compare to theory:
fname = joinpath(pwd(), "results", "plot.dat")
data = DelimitedFiles.readdlm(fname, comments=true)
data = transpose(data)
z = data[1, :]   # [zu] propagation distance
I = data[2, :]   # [Iu] itensity
tau = data[4, :] * 1e-15   # [m] 1/e pulse duration

k2 = 2.131584978375e-29   # [s^2/m] - 2nd derivative of wave number: d(k1)/dw
tau0 = 20e-15  # [s] initial pulse duration
zd = tau0^2 / k2
tauth = @. tau0 * sqrt(1 + (z / zd)^2)   # [m] theoretical 1/e pulse duration
Ith = @. tau0 / tauth   # [I0] theoretical intensity

@test maximum(abs.(tau .- tauth)) <= 0.1e-15   # difference is less than the step size
@test maximum(abs.(I .- Ith)) <= 1e-3   # difference is less than 0.1%

# Delete output files:
rm("results", recursive=true)

cd(here)

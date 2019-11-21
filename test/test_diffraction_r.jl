import DelimitedFiles
using PyPlot
PROJECT = joinpath(@__DIR__, "..")
function main()



# Run simulations:
cd(joinpath(PROJECT, "test", "diffraction", "R"))
proc = run(`julia -O3 --check-bounds=no $PROJECT/src/Filament.jl input.jl`)
#@test proc.exitcode == 0

# Compare to theory:
fname = joinpath(pwd(), "results", "plot.dat")
data = DelimitedFiles.readdlm(fname, comments=true)
data = transpose(data)
z = data[1, :]   # [zu] propagation distance
I = data[2, :]   # [Iu] itensity
a = data[3, :] * 1e-3   # [m] 1/e beam radius

lam0 = 800e-9   # [m] central wavelength
a0 = 1e-3   # [m] initial beam radius
zd = 2 * pi / lam0 * a0^2   # [m] diffraction length
ath = @. a0 * sqrt(1 + (z / zd)^2)   # [m] theoretical 1/e beam radius
Ith = @. (a0 / ath)^2   # [I0] theoretical intensity

@show maximum(abs.(a .- ath)) <= 0.02e-3   # difference is less than the step size
@show maximum(abs.(I .- Ith)) <= 1e-3   # difference is less than 0.1%

# Delete the directory with the results of simulation:
# rm(joinpath(pwd(), "results"), recursive=true)


subplot(211)
plot(z, Ith)
plot(z, I)
subplot(212)
plot(z, ath)
plot(z, a)
show()
end
main()

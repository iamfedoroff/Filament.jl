using PyCall
@pyimport numpy.fft as npfft
@pyimport matplotlib.pyplot as plt
@pyimport scipy.constants as sc

push!(LOAD_PATH, joinpath(@__DIR__, "..", "modules"))
import Units
import Grids
import Fields
import Media
import Models
import Infos
import Plots

C0 = sc.c   # speed of light in vacuum


# Read input file and change current working directory
file_input = abspath(ARGS[1])
include(file_input)
cd(dirname(file_input))


# Prepare units and grid
unit = Units.Unit(ru, zu, tu, Iu, rhou)
grid = Grids.Grid(rmax, Nr, tmin, tmax, Nt)


# Read the initial condition file and prepare field
include(abspath(file_initial_condition))
z = z / unit.z   # convert initial z to dimensionless units
field = Fields.Field(unit, grid, lam0, initial_condition)


# Read the medium file and prepare medium
include(abspath(file_medium))
medium = Media.Medium(permittivity, permeability, n2)


# Prepare output files
prefix_dir = dirname(prefix)
prefix_name = basename(prefix)

if prefix_dir != ""
    mkpath(prefix_dir)
end

file_infos = joinpath(prefix_dir, string(prefix_name, "info.txt"))
info = Infos.Info(file_infos, file_input, file_initial_condition, file_medium,
                  unit, grid, medium, field)

file_plotdat = joinpath(prefix_dir, string(prefix_name, "plot.dat"))
plotdat = Plots.PlotDAT(file_plotdat, unit)
Plots.writeDAT(plotdat, z, field)

file_plothdf = joinpath(prefix_dir, string(prefix_name, "plot.h5"))
plothdf = Plots.PlotHDF(file_plothdf, unit, grid)
Plots.writeHDF(plothdf, z, field)
Plots.writeHDF_zdata(plothdf, z, field)


# Prepare model
keys = Dict()

model = Models.Model(unit, grid, field, medium, keys)

quit()


# unshift!(PyVector(pyimport("sys")["path"]), "/home/fedoroff/storage/projects/Filament/jlFilament/modules/")
# @pyimport units
# unit = units.UnitRT(ru, zu, tu, Iu, rhou)


# F = Fields.fluence(field)

# plt.figure(dpi=300)
# # plt.plot(grid.r, field.E[:, div(grid.Nt, 2)])
# # plt.plot(grid.t, real(field.E[1, :]))
# plt.plot(grid.r, F)
# plt.tight_layout()
# plt.show()
# quit()








# ******************************************************************************
# Propagation
# ******************************************************************************
z = Media.diffraction_length(medium, field.w0, a0)

I0 = abs2.(field.E[:, div(grid.Ny, 2)])

Models.zstep(z, field, model)

Iz = abs2.(field.E[:, div(grid.Ny, 2)])

# ******************************************************************************
# Plot
# ******************************************************************************
plt.figure(dpi=300)
plt.plot(grid.x, I0)
plt.plot(grid.x, Iz)
plt.tight_layout()
plt.show()

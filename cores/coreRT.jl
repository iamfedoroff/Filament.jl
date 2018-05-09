using PyCall
@pyimport numpy.fft as npfft
@pyimport matplotlib.pyplot as plt
@pyimport scipy.constants as sc

# append the home directory of the project to the search path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "modules"))
import Units
import Grids
import Fields
import Media
import Models
import Infos

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
z = z / unit.z   # normalize initial z
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
fname_plotdat = joinpath(prefix_dir, string(prefix_name, "plot.dat"))
fname_plothdf = joinpath(prefix_dir, string(prefix_name, "plot.h5"))

info = Infos.Info(file_infos, file_input, file_initial_condition, file_medium,
                  unit, grid, medium, field)

# Infos.write_message(info, "Hello\n")
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






# Prepare model
# model = Models.Model(unit, grid, field, medium)

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

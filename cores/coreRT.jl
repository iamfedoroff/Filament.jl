using PyCall
@pyimport numpy.fft as npfft
@pyimport matplotlib.pyplot as plt
@pyimport scipy.constants as sc

import Formatting

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
keys = Dict(
    "KPARAXIAL" => KPARAXIAL,
    "rguard_width" => rguard_width, "tguard_width" => tguard_width,
    "kguard" => kguard, "wguard" => wguard
    )

model = Models.Model(unit, field, medium, keys)


# Main loop
stime = now()

znext_plothdf = z + dz_plothdf

dz_zdata = 0.5 * field.lam0
znext_zdata = z + dz_zdata

@timev while z < zmax
    Imax = Fields.peak_intensity(field)
    rhomax = Fields.peak_plasma_density(field)

    # Adaptive z step
    # dz = model.adaptive_dz(dzAdaptLevel, Imax, Nemax)
    # dz = min(dz_initial, dz_plothdf, dz)
    dz = min(dz_initial, dz_plothdf)
    z = z + dz

    print("z=$(Formatting.fmt("18.12e", z))[zu] " *
          "I=$(Formatting.fmt("18.12e", Imax))[Iu] " *
          "rho=$(Formatting.fmt("18.12e", rhomax))[rhou]\n")

    Models.zstep(dz, field, model)

    # Plots
    Plots.writeDAT(plotdat, z, field)   # write to plotdat file

    if z >= znext_plothdf
        Plots.writeHDF(plothdf, z, field)   # write to plothdf file
        znext_plothdf = znext_plothdf + dz_plothdf
    end

    if z >= znext_zdata
        Plots.writeHDF_zdata(plothdf, z, field)  # write 1d data to plothdf file
        znext_zdata = z + dz_zdata
    end

    # Exit conditions
    if Imax > Istop
        Plots.writeHDF_zdata(plothdf, z, field)
        message = "Stop (Imax >= Istop): z=$(z)[zu], z=$(z * unit.z)[m]"
        Infos.write_message(info, message)
        break
    end

end

etime = now()
ttime = Dates.canonicalize(Dates.CompoundPeriod(etime - stime))
message = "Start time: $(stime)\n" *
          "End time:   $(etime)\n" *
          "Run time:   $(ttime)\n"
Infos.write_message(info, message)

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

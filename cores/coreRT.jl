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
import Plasmas
import Models
import Infos
import WritePlots

const C0 = sc.c   # speed of light in vacuum

# ******************************************************************************
# Read input file and change current working directory
# ******************************************************************************
file_input = abspath(ARGS[1])
include(file_input)
cd(dirname(file_input))

# ******************************************************************************
# Prepare units and grid
# ******************************************************************************
unit = Units.Unit(ru, zu, tu, Iu, rhou)
grid = Grids.Grid(rmax, Nr, tmin, tmax, Nt)

# ******************************************************************************
# Read the initial condition file and prepare field
# ******************************************************************************
include(abspath(file_initial_condition))
z = z / unit.z   # convert initial z to dimensionless units
field = Fields.Field(unit, grid, lam0, initial_condition)

# ******************************************************************************
# Read the medium file and prepare medium and plasma
# ******************************************************************************
include(abspath(file_medium))
medium = Media.Medium(permittivity, permeability, n2, rho0, nuc, mr)

keys = Dict("IONARG" => IONARG, "AVALANCHE" => AVALANCHE)
plasma = Plasmas.Plasma(unit, grid, field, medium, rho0, components, keys)
Plasmas.free_charge(plasma, grid, field)

# ******************************************************************************
# Prepare output files
# ******************************************************************************
prefix_dir = dirname(prefix)
prefix_name = basename(prefix)

if prefix_dir != ""
    mkpath(prefix_dir)
end

file_infos = joinpath(prefix_dir, string(prefix_name, "info.txt"))
info = Infos.Info(file_infos, file_input, file_initial_condition, file_medium,
                  unit, grid, field, medium, plasma)

file_plotdat = joinpath(prefix_dir, string(prefix_name, "plot.dat"))
plotdat = WritePlots.PlotDAT(file_plotdat, unit)
WritePlots.writeDAT(plotdat, z, field)

file_plothdf = joinpath(prefix_dir, string(prefix_name, "plot.h5"))
plothdf = WritePlots.PlotHDF(file_plothdf, unit, grid)
WritePlots.writeHDF(plothdf, z, field)
WritePlots.writeHDF_zdata(plothdf, z, field)

# ******************************************************************************
# Prepare model
# ******************************************************************************
keys = Dict("KPARAXIAL" => KPARAXIAL, "QPARAXIAL" => QPARAXIAL, "KERR" => KERR,
            "THG" => THG, "PLASMA" => PLASMA, "ILOSSES" => ILOSSES,
            "IONARG" => IONARG, "rguard_width" => rguard_width,
            "tguard_width" => tguard_width, "kguard" => kguard,
            "wguard" => wguard)
model = Models.Model(unit, grid, field, medium, keys)

# ******************************************************************************
# Main loop
# ******************************************************************************
stime = now()

znext_plothdf = z + dz_plothdf

dz_zdata = 0.5 * field.lam0
znext_zdata = z + dz_zdata

@time while z < zmax
    Imax = Fields.peak_intensity(field)
    rhomax = Fields.peak_plasma_density(field)

    # Adaptive z step
    dz = Models.adaptive_dz(model, dzAdaptLevel, Imax, rhomax)
    dz = min(dz_initial, dz_plothdf, dz)
    z = z + dz

    print("z=$(Formatting.fmt("18.12e", z))[zu] " *
          "I=$(Formatting.fmt("18.12e", Imax))[Iu] " *
          "rho=$(Formatting.fmt("18.12e", rhomax))[rhou]\n")

    Models.zstep(dz, grid, field, plasma, model)

    # Write integral parameters to dat file
    WritePlots.writeDAT(plotdat, z, field)

    # Write field to hdf file
    if z >= znext_plothdf
        WritePlots.writeHDF(plothdf, z, field)
        znext_plothdf = znext_plothdf + dz_plothdf
    end

    # Write 1d field data to hdf file
    if z >= znext_zdata
        WritePlots.writeHDF_zdata(plothdf, z, field)
        znext_zdata = z + dz_zdata
    end

    # Exit conditions
    if Imax > Istop
        WritePlots.writeHDF_zdata(plothdf, z, field)
        message = "Stop (Imax >= Istop): z=$(z)[zu], z=$(z * unit.z)[m]\n"
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

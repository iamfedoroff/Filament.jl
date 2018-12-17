module WritePlots

import Formatting
import HDF5

import Units
import Grids
import Fields


# ******************************************************************************
# PlotDAT
# ******************************************************************************
struct PlotVar
    name :: String
    siunit :: String
    unit :: Float64
    func :: Function
end


struct PlotDAT
    fname :: String
    plotvars :: Array{PlotVar, 1}
end


function PlotDAT(fname::String, unit::Units.Unit)
    plotvars = [
        PlotVar("Fmax", "J/m^2", unit.t * unit.I, Fields.peak_fluence),
        PlotVar("Imax", "W/m2", unit.I, Fields.peak_intensity),
        PlotVar("Nemax", "1/m3", unit.rho, Fields.peak_plasma_density),
        PlotVar("De", "1/m", unit.r^2 * unit.rho, Fields.linear_plasma_density),
        PlotVar("rfil", "m", unit.r, Fields.beam_radius),
        PlotVar("rpl", "m", unit.r, Fields.plasma_radius),
        PlotVar("W", "J", unit.I * unit.t * unit.r^2, Fields.energy),
        ]

    # Write header:
    fp = open(fname, "w")

    # write names:
    write(fp, "#")
    write(fp, " $(Formatting.fmt("<18", "z"))")
    for plotvar in plotvars
        write(fp, " $(Formatting.fmt("<18", plotvar.name))")
    end
    write(fp, "\n")

    # write SI units:
    write(fp, "#")
    write(fp, " $(Formatting.fmt("<18", "m"))")
    for plotvar in plotvars
        write(fp, " $(Formatting.fmt("<18", plotvar.siunit))")
    end
    write(fp, "\n")

    # write dimensionless units:
    write(fp, "#")
    write(fp, " $(Formatting.fmt("<18", unit.z))")
    for plotvar in plotvars
        write(fp, " $(Formatting.fmt("<18", plotvar.unit))")
    end
    write(fp, "\n")

    close(fp)

    return PlotDAT(fname, plotvars)
end


function writeDAT(plotdat::PlotDAT, z::Float64, field::Fields.Field)
    fp = open(plotdat.fname, "a")
    write(fp, "  $(Formatting.fmt("18.12e", z)) ")
    for plotvar in plotdat.plotvars
        var = plotvar.func(field)
        write(fp, "$(Formatting.fmt("18.12e", var)) ")
    end
    write(fp, "\n")
    close(fp)
end


# ******************************************************************************
# PlotHDF
# ******************************************************************************
VERSION = 2.0
GROUP_UNIT = "units"
GROUP_GRID = "grid"
GROUP_FDAT = "field"
GROUP_ZDAT = "zdata"


mutable struct PlotHDF
    fname :: String
    numplot :: Int64
    previous_z :: Float64
    iz :: Int64
end


function PlotHDF(fname::String, unit::Units.Unit, grid::Grids.Grid)
    fp = HDF5.h5open(fname, "w")

    HDF5.g_create(fp, GROUP_UNIT)
    HDF5.g_create(fp, GROUP_GRID)
    HDF5.g_create(fp, GROUP_FDAT)
    HDF5.g_create(fp, GROUP_ZDAT)

    fp["version"] = VERSION

    group_unit = fp[GROUP_UNIT]
    group_unit["r"] = unit.r
    group_unit["z"] = unit.z
    group_unit["t"] = unit.t
    group_unit["I"] = unit.I
    group_unit["Ne"] = unit.rho

    group_grid = fp[GROUP_GRID]
    group_grid["geometry"] = grid.geometry
    group_grid["rmax"] = grid.rmax
    group_grid["Nr"] = grid.Nr
    group_grid["tmin"] = grid.tmin
    group_grid["tmax"] = grid.tmax
    group_grid["Nt"] = grid.Nt

    HDF5.close(fp)

    numplot = 0
    previous_z = -Inf
    iz = 0

    return PlotHDF(fname, numplot, previous_z, iz)
end


function writeHDF(plothdf::PlotHDF, z::Float64, field::Fields.Field)
    if z == plothdf.previous_z
        return
    end

    dset = "$(Formatting.fmt("03d", plothdf.numplot))"
    print(" Writing dataset $(dset)...\n")

    plothdf.numplot = plothdf.numplot + 1
    plothdf.previous_z = z

    fp = HDF5.h5open(plothdf.fname, "r+")
    group_fdat = fp[GROUP_FDAT]
    # group_fdat[dset] = real(field.E)
    group_fdat[dset] = collect(transpose(real(field.E)))
    HDF5.attrs(group_fdat[dset])["z"] = z
    HDF5.close(fp)
end


function writeHDF_zdata(plothdf::PlotHDF, z::Float64, field::Fields.Field)
    plothdf.iz = plothdf.iz + 1
    iz = plothdf.iz

    fp = HDF5.h5open(plothdf.fname, "r+")
    group_zdat = fp[GROUP_ZDAT]

    data_name = "z"
    if ! HDF5.exists(group_zdat, data_name)
        HDF5.d_create(group_zdat, data_name, Float64,
                      ((1,), (-1,)), "chunk", (1,))
    end
    data = group_zdat[data_name]
    HDF5.set_dims!(data, (iz,))
    data[iz] = z

    data_name = "Fzx"
    if ! HDF5.exists(group_zdat, data_name)
        HDF5.d_create(group_zdat, data_name, Float64,
                      # ((1, field.grid.Nr), (-1, field.grid.Nr)),
                      ((field.grid.Nr, 1), (field.grid.Nr, -1)),
                      "chunk", (1, 1))
    end
    data = group_zdat[data_name]
    # HDF5.set_dims!(data, (iz, field.grid.Nr))
    # data[iz, :] = Fields.fluence(field)
    HDF5.set_dims!(data, (field.grid.Nr, iz))
    data[:, iz] = Fields.fluence(field)

    data_name = "Nezx"
    if ! HDF5.exists(group_zdat, data_name)
        HDF5.d_create(group_zdat, data_name, Float64,
                      ((1, field.grid.Nr), (-1, field.grid.Nr)),
                      "chunk", (1, 1))
    end
    data = group_zdat[data_name]
    HDF5.set_dims!(data, (iz, field.grid.Nr))
    data[iz, :] = field.rho

    data_name = "iSzf"
    if ! HDF5.exists(group_zdat, data_name)
        HDF5.d_create(group_zdat, data_name, Float64,
                      ((1, field.grid.Nw), (-1, field.grid.Nw)),
                      "chunk", (1, 1))
    end
    data = group_zdat[data_name]
    HDF5.set_dims!(data, (iz, field.grid.Nw))
    data[iz, :] = Fields.integral_power_spectrum(field)

    HDF5.close(fp)
end


end

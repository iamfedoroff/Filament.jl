module WritePlots

import Formatting
import CuArrays
import HDF5

import PyCall

import Units
import Grids
import Fields
import Plasmas

const FloatGPU = Float32


# ******************************************************************************
# PlotDAT
# ******************************************************************************
struct PlotVar
    name :: String
    siunit :: String
    unit :: Float64
end


struct PlotDAT
    fname :: String
    plotvars :: Array{PlotVar, 1}
end


function PlotDAT(fname::String, unit::Units.Unit)
    plotvars = [
        PlotVar("Fmax", "J/m^2", unit.t * unit.I),
        PlotVar("Imax", "W/m2", unit.I),
        PlotVar("Nemax", "1/m3", unit.rho),
        PlotVar("De", "1/m", unit.r^2 * unit.rho),
        PlotVar("rfil", "m", unit.r),
        PlotVar("rpl", "m", unit.r),
        PlotVar("W", "J", unit.I * unit.t * unit.r^2),
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


function writeDAT(plotdat::PlotDAT, z::Float64, grid::Grids.Grid,
                  field::Fields.Field, plasma::Plasmas.Plasma)
    fp = open(plotdat.fname, "a")

    # z
    write(fp, "  $(Formatting.fmt("18.12e", z)) ")

    # Fmax
    var = Fields.peak_fluence(grid, field)
    write(fp, "$(Formatting.fmt("18.12e", var)) ")

    # Imax
    var = Fields.peak_intensity(field)
    write(fp, "$(Formatting.fmt("18.12e", var)) ")

    # Nemax
    var = Plasmas.peak_plasma_density(plasma)
    write(fp, "$(Formatting.fmt("18.12e", var)) ")

    # De
    var = Plasmas.linear_plasma_density(grid, plasma)
    write(fp, "$(Formatting.fmt("18.12e", var)) ")

    # rfil
    var = Fields.beam_radius(grid, field)
    write(fp, "$(Formatting.fmt("18.12e", var)) ")

    # rpl
    var = Plasmas.plasma_radius(grid, plasma)
    write(fp, "$(Formatting.fmt("18.12e", var)) ")

    # W
    var = Fields.energy(grid, field)
    write(fp, "$(Formatting.fmt("18.12e", var)) ")

    write(fp, "\n")
    close(fp)
end


# ******************************************************************************
# PlotHDF
# ******************************************************************************
const PFVERSION = "2.0"
const GROUP_UNIT = "units"
const GROUP_GRID = "grid"
const GROUP_FDAT = "field"
const GROUP_ZDAT = "zdata"


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

    # fp["version"] = PFVERSION

    group_unit = fp[GROUP_UNIT]
    group_unit["r"] = unit.r
    group_unit["z"] = unit.z
    group_unit["t"] = unit.t
    group_unit["I"] = unit.I
    group_unit["Ne"] = unit.rho

    group_grid = fp[GROUP_GRID]
    # group_grid["geometry"] = grid.geometry
    group_grid["rmax"] = grid.rmax
    group_grid["Nr"] = grid.Nr
    group_grid["tmin"] = grid.tmin
    group_grid["tmax"] = grid.tmax
    group_grid["Nt"] = grid.Nt

    HDF5.close(fp)

    # Write version and grid geometry which are compatable with h5py:
    h5py = PyCall.pyimport("h5py")
    fp = h5py.File(fname, "a")
    fp.create_dataset("version", data=PFVERSION)
    fp.create_dataset(GROUP_GRID * "/geometry", data=grid.geometry)
    fp.close()

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
    # group_fdat[dset] = CuArrays.collect(real.(field.E))
    group_fdat[dset] = CuArrays.collect(transpose(real.(field.E)))
    HDF5.attrs(group_fdat[dset])["z"] = z
    HDF5.close(fp)
end


function writeHDF_zdata(plothdf::PlotHDF, z::Float64, grid::Grids.Grid,
                        field::Fields.Field, plasma::Plasmas.Plasma)
    plothdf.iz = plothdf.iz + 1
    iz = plothdf.iz

    fp = HDF5.h5open(plothdf.fname, "r+")
    group_zdat = fp[GROUP_ZDAT]

    data_name = "z"
    if ! HDF5.exists(group_zdat, data_name)
        HDF5.d_create(group_zdat, data_name, FloatGPU,
                      ((1,), (-1,)), "chunk", (1,))
    end
    data = group_zdat[data_name]
    HDF5.set_dims!(data, (iz,))
    data[iz] = FloatGPU(z)

    data_name = "Fzx"
    if ! HDF5.exists(group_zdat, data_name)
        HDF5.d_create(group_zdat, data_name, FloatGPU,
                      # ((1, grid.Nr), (-1, grid.Nr)),
                      ((grid.Nr, 1), (grid.Nr, -1)),
                      "chunk", (1, 1))
    end
    data = group_zdat[data_name]
    # HDF5.set_dims!(data, (iz, grid.Nr))
    # data[iz, :] = Fields.fluence(grid, field)
    HDF5.set_dims!(data, (grid.Nr, iz))
    data[:, iz] = Fields.fluence(grid, field)

    data_name = "Nezx"
    if ! HDF5.exists(group_zdat, data_name)
        HDF5.d_create(group_zdat, data_name, FloatGPU,
                      # ((1, grid.Nr), (-1, grid.Nr)),
                      ((grid.Nr, 1), (grid.Nr, -1)),
                      "chunk", (1, 1))
    end
    data = group_zdat[data_name]
    # HDF5.set_dims!(data, (iz, grid.Nr))
    # data[iz, :] = plasma.rho_end
    HDF5.set_dims!(data, (grid.Nr, iz))
    data[:, iz] = plasma.rho_end

    data_name = "iSzf"
    if ! HDF5.exists(group_zdat, data_name)
        HDF5.d_create(group_zdat, data_name, FloatGPU,
                      # ((1, grid.Nw), (-1, grid.Nw)),
                      ((grid.Nw, 1), (grid.Nw, -1)),
                      "chunk", (1, 1))
    end
    data = group_zdat[data_name]
    # HDF5.set_dims!(data, (iz, grid.Nw))
    # data[iz, :] = Fields.integral_power_spectrum(grid, field)
    HDF5.set_dims!(data, (grid.Nw, iz))
    data[:, iz] = Fields.integral_power_spectrum(grid, field)

    HDF5.close(fp)
end


end

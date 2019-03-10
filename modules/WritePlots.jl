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
# PlotCache
# ******************************************************************************
mutable struct PlotCache
    Fmax :: Float64
    Imax :: Float64
    rhomax :: Float64
    De :: Float64
    rfil :: Float64
    rpl :: Float64
    W :: Float64
    I :: CuArrays.CuArray{FloatGPU, 2}
    F :: Array{FloatGPU, 1}
    rho :: Array{FloatGPU, 1}
    S :: Array{FloatGPU, 1}
end


function PlotCache(grid::Grids.Grid)
    I = CuArrays.cuzeros((grid.Nr, grid.Nt))
    F = zeros(FloatGPU, grid.Nr)
    rho = zeros(FloatGPU, grid.Nr)
    S = zeros(FloatGPU, grid.Nw)
    return PlotCache(0., 0., 0., 0., 0., 0., 0., I, F, rho, S)
end


function plotcache_update!(pcache::PlotCache, grid::Grids.Grid,
                           field::Fields.Field, plasma::Plasmas.Plasma)
    pcache.I .= abs2.(field.E)

    F = sum(pcache.I .* FloatGPU(grid.dt), dims=2)
    pcache.F[:] = CuArrays.collect(F)[:, 1]

    rho = plasma.rho[:, end]
    pcache.rho[:] = CuArrays.collect(rho)

    pcache.Fmax = Float64(maximum(F))
    pcache.Imax = Float64(maximum(pcache.I))
    pcache.rhomax = Float64(maximum(rho))
    pcache.De = sum(rho .* grid.rdr) * 2. * pi
    pcache.rfil = 2. * Fields.radius(grid.r, pcache.F)
    pcache.rpl = 2. * Fields.radius(grid.r, pcache.rho)
    pcache.W = sum(F .* grid.rdr) * 2. * pi
    pcache.S = Fields.integral_power_spectrum(grid, field)
    return nothing
end


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


function writeDAT(plotdat::PlotDAT, z::Float64, pcache::PlotCache)
    fp = open(plotdat.fname, "a")
    write(fp, "  $(Formatting.fmt("18.12e", z)) ")   # z
    write(fp, "$(Formatting.fmt("18.12e", pcache.Fmax)) ")   # Fmax
    write(fp, "$(Formatting.fmt("18.12e", pcache.Imax)) ")   # Imax
    write(fp, "$(Formatting.fmt("18.12e", pcache.rhomax)) ")   # Nemax
    write(fp, "$(Formatting.fmt("18.12e", pcache.De)) ")   # De
    write(fp, "$(Formatting.fmt("18.12e", pcache.rfil)) ")   # rfil
    write(fp, "$(Formatting.fmt("18.12e", pcache.rpl)) ")   # rpl
    write(fp, "$(Formatting.fmt("18.12e", pcache.W)) ")   # W
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

    group_zdat = fp[GROUP_ZDAT]
    HDF5.d_create(group_zdat, "z", FloatGPU, ((1,), (-1,)), "chunk", (1,))
    HDF5.d_create(group_zdat, "Fzx", FloatGPU,
                  # ((1, grid.Nr), (-1, grid.Nr)),
                  ((grid.Nr, 1), (grid.Nr, -1)),
                  "chunk", (1, 1))
    HDF5.d_create(group_zdat, "Nezx", FloatGPU,
                  # ((1, grid.Nr), (-1, grid.Nr)),
                  ((grid.Nr, 1), (grid.Nr, -1)),
                  "chunk", (1, 1))
    HDF5.d_create(group_zdat, "iSzf", FloatGPU,
                  # ((1, grid.Nw), (-1, grid.Nw)),
                  ((grid.Nw, 1), (grid.Nw, -1)),
                  "chunk", (1, 1))

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


function writeHDF_zdata(plothdf::PlotHDF, z::Float64, pcache::PlotCache)
    plothdf.iz = plothdf.iz + 1
    iz = plothdf.iz

    fp = HDF5.h5open(plothdf.fname, "r+")
    
    group_zdat = fp[GROUP_ZDAT]

    data = group_zdat["z"]
    HDF5.set_dims!(data, (iz,))
    data[iz] = FloatGPU(z)

    data = group_zdat["Fzx"]
    # HDF5.set_dims!(data, (iz, grid.Nr))
    # data[iz, :] = pcache.F
    HDF5.set_dims!(data, (length(pcache.F), iz))
    data[:, iz] = pcache.F

    data = group_zdat["Nezx"]
    # HDF5.set_dims!(data, (iz, grid.Nr))
    # data[iz, :] = pcache.rho
    HDF5.set_dims!(data, (length(pcache.rho), iz))
    data[:, iz] = pcache.rho

    data = group_zdat["iSzf"]
    # HDF5.set_dims!(data, (iz, grid.Nw))
    # data[iz, :] = pcache.S
    HDF5.set_dims!(data, (length(pcache.S), iz))
    data[:, iz] = pcache.S

    HDF5.close(fp)
end


end

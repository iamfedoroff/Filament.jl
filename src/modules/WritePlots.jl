module WritePlots

import Formatting
import CuArrays
import HDF5

import PyCall

import Units
import Grids
import Fields

const FloatGPU = Float32
const ComplexGPU = ComplexF32


struct PlotVar
    name :: String
    siunit :: String
    unit :: Float64
end


# ******************************************************************************
# PlotVarData
# ******************************************************************************
abstract type PlotVarData end


mutable struct PlotVarDataR <: PlotVarData
    plotvars :: Array{PlotVar, 1}
    Imax :: Float64
    rfil :: Float64
    P :: Float64
    I :: Array{FloatGPU, 1}
    Igpu :: CuArrays.CuArray{FloatGPU, 1}
end


mutable struct PlotVarDataRT <: PlotVarData
    plotvars :: Array{PlotVar, 1}
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


mutable struct PlotVarDataXY <: PlotVarData
    plotvars :: Array{PlotVar, 1}
    Imax :: Float64
    ax :: Float64
    ay :: Float64
    P :: Float64
    I :: Array{FloatGPU, 2}
    Igpu :: CuArrays.CuArray{FloatGPU, 2}
end


function PlotVarData(unit::Units.UnitR, grid::Grids.GridR)
    plotvars = [
        PlotVar("Imax", "W/m2", unit.I),
        PlotVar("rfil", "m", unit.r),
        PlotVar("P", "W", unit.r^2 * unit.I),
        ]
    I = zeros(FloatGPU, grid.Nr)
    Igpu = CuArrays.zeros(FloatGPU, grid.Nr)
    return PlotVarDataR(plotvars, 0., 0., 0., I, Igpu)
end


function PlotVarData(unit::Units.UnitRT, grid::Grids.GridRT)
    plotvars = [
        PlotVar("Fmax", "J/m^2", unit.t * unit.I),
        PlotVar("Imax", "W/m2", unit.I),
        PlotVar("Nemax", "1/m3", unit.rho),
        PlotVar("De", "1/m", unit.r^2 * unit.rho),
        PlotVar("rfil", "m", unit.r),
        PlotVar("rpl", "m", unit.r),
        PlotVar("W", "J", unit.r^2 * unit.t * unit.I),
        ]
    I = CuArrays.zeros(FloatGPU, (grid.Nr, grid.Nt))
    F = zeros(FloatGPU, grid.Nr)
    rho = zeros(FloatGPU, grid.Nr)
    S = zeros(FloatGPU, grid.Nw)
    return PlotVarDataRT(plotvars, 0., 0., 0., 0., 0., 0., 0., I, F, rho, S)
end


function PlotVarData(unit::Units.UnitXY, grid::Grids.GridXY)
    plotvars = [
        PlotVar("Imax", "W/m2", unit.I),
        PlotVar("ax", "m", unit.x),
        PlotVar("ay", "m", unit.y),
        PlotVar("P", "W", unit.x * unit.y * unit.I),
        ]
    I = zeros(FloatGPU, (grid.Nx, grid.Ny))
    Igpu = CuArrays.zeros(FloatGPU, (grid.Nx, grid.Ny))
    return PlotVarDataXY(plotvars, 0., 0., 0., 0., I, Igpu)
end


function pdata_update!(pdata::PlotVarDataR, grid::Grids.GridR,
                       field::Fields.FieldR)
    pdata.Igpu .= abs2.(field.E)
    pdata.I[:] = CuArrays.collect(pdata.Igpu)

    pdata.Imax = Float64(maximum(pdata.Igpu))
    pdata.rfil = 2. * Grids.radius(grid.r, pdata.I)
    pdata.P = sum(pdata.Igpu .* grid.rdr) * 2. * pi
    return nothing
end


function pdata_update!(pdata::PlotVarDataRT, grid::Grids.GridRT,
                       field::Fields.FieldRT)
    pdata.I .= abs2.(field.E)

    F = sum(pdata.I .* FloatGPU(grid.dt), dims=2)
    pdata.F[:] = CuArrays.collect(F)[:, 1]

    rho = field.rho[:, end]
    pdata.rho[:] = CuArrays.collect(rho)

    pdata.Fmax = Float64(maximum(F))
    pdata.Imax = Float64(maximum(pdata.I))
    pdata.rhomax = Float64(maximum(rho))
    pdata.De = sum(rho .* grid.rdr) * 2. * pi
    pdata.rfil = 2. * Grids.radius(grid.r, pdata.F)
    pdata.rpl = 2. * Grids.radius(grid.r, pdata.rho)
    pdata.W = sum(F .* grid.rdr) * 2. * pi
    pdata.S = Fields.integral_power_spectrum(grid, field)
    return nothing
end


function pdata_update!(pdata::PlotVarDataXY, grid::Grids.GridXY,
                       field::Fields.FieldXY)
    pdata.Igpu .= abs2.(field.E)
    pdata.I[:] = CuArrays.collect(pdata.Igpu)

    pdata.Imax, imax = findmax(pdata.I)
    pdata.ax = Grids.radius(grid.x, pdata.I[:, imax[2]])
    pdata.ay = Grids.radius(grid.y, pdata.I[imax[1], :])
    pdata.P = sum(pdata.Igpu) * grid.dx * grid.dy
    return nothing
end


# ******************************************************************************
# PlotDAT
# ******************************************************************************
struct PlotDAT
    fname :: String
end


function PlotDAT(fname::String, unit::Units.Unit, pdata::PlotVarData)
    fp = open(fname, "w")

    # write names:
    write(fp, "#")
    write(fp, " $(Formatting.fmt("<18", "z"))")
    for plotvar in pdata.plotvars
        write(fp, " $(Formatting.fmt("<18", plotvar.name))")
    end
    write(fp, "\n")

    # write SI units:
    write(fp, "#")
    write(fp, " $(Formatting.fmt("<18", "m"))")
    for plotvar in pdata.plotvars
        write(fp, " $(Formatting.fmt("<18", plotvar.siunit))")
    end
    write(fp, "\n")

    # write dimensionless units:
    write(fp, "#")
    write(fp, " $(Formatting.fmt("<18", unit.z))")
    for plotvar in pdata.plotvars
        write(fp, " $(Formatting.fmt("<18", plotvar.unit))")
    end
    write(fp, "\n")

    close(fp)

    return PlotDAT(fname)
end


function writeDAT(plotdat::PlotDAT, z::Float64, pdata::PlotVarDataR)
    fp = open(plotdat.fname, "a")
    write(fp, "  $(Formatting.fmt("18.12e", z)) ")   # z
    write(fp, "$(Formatting.fmt("18.12e", pdata.Imax)) ")   # Imax
    write(fp, "$(Formatting.fmt("18.12e", pdata.rfil)) ")   # rfil
    write(fp, "$(Formatting.fmt("18.12e", pdata.P)) ")   # P
    write(fp, "\n")
    close(fp)
end


function writeDAT(plotdat::PlotDAT, z::Float64, pdata::PlotVarDataRT)
    fp = open(plotdat.fname, "a")
    write(fp, "  $(Formatting.fmt("18.12e", z)) ")   # z
    write(fp, "$(Formatting.fmt("18.12e", pdata.Fmax)) ")   # Fmax
    write(fp, "$(Formatting.fmt("18.12e", pdata.Imax)) ")   # Imax
    write(fp, "$(Formatting.fmt("18.12e", pdata.rhomax)) ")   # Nemax
    write(fp, "$(Formatting.fmt("18.12e", pdata.De)) ")   # De
    write(fp, "$(Formatting.fmt("18.12e", pdata.rfil)) ")   # rfil
    write(fp, "$(Formatting.fmt("18.12e", pdata.rpl)) ")   # rpl
    write(fp, "$(Formatting.fmt("18.12e", pdata.W)) ")   # W
    write(fp, "\n")
    close(fp)
end


function writeDAT(plotdat::PlotDAT, z::Float64, pdata::PlotVarDataXY)
    fp = open(plotdat.fname, "a")
    write(fp, "  $(Formatting.fmt("18.12e", z)) ")   # z
    write(fp, "$(Formatting.fmt("18.12e", pdata.Imax)) ")   # Imax
    write(fp, "$(Formatting.fmt("18.12e", pdata.ax)) ")   # ax
    write(fp, "$(Formatting.fmt("18.12e", pdata.ay)) ")   # ay
    write(fp, "$(Formatting.fmt("18.12e", pdata.P)) ")   # P
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


function PlotHDF(fname::String, unit::Units.UnitR, grid::Grids.GridR)
    fp = HDF5.h5open(fname, "w")

    HDF5.g_create(fp, GROUP_UNIT)
    HDF5.g_create(fp, GROUP_GRID)
    HDF5.g_create(fp, GROUP_FDAT)
    HDF5.g_create(fp, GROUP_ZDAT)

    # fp["version"] = PFVERSION

    group_unit = fp[GROUP_UNIT]
    group_unit["r"] = unit.r
    group_unit["z"] = unit.z
    group_unit["I"] = unit.I

    group_grid = fp[GROUP_GRID]
    # group_grid["geometry"] = grid.geometry
    group_grid["rmax"] = grid.rmax
    group_grid["Nr"] = grid.Nr

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


function PlotHDF(fname::String, unit::Units.UnitRT, grid::Grids.GridRT)
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


function PlotHDF(fname::String, unit::Units.UnitXY, grid::Grids.GridXY)
    fp = HDF5.h5open(fname, "w")

    HDF5.g_create(fp, GROUP_UNIT)
    HDF5.g_create(fp, GROUP_GRID)
    HDF5.g_create(fp, GROUP_FDAT)
    HDF5.g_create(fp, GROUP_ZDAT)

    # fp["version"] = PFVERSION

    group_unit = fp[GROUP_UNIT]
    group_unit["x"] = unit.x
    group_unit["y"] = unit.y
    group_unit["z"] = unit.z
    group_unit["I"] = unit.I

    group_grid = fp[GROUP_GRID]
    # group_grid["geometry"] = grid.geometry
    group_grid["xmin"] = grid.xmin
    group_grid["xmax"] = grid.xmax
    group_grid["Nx"] = grid.Nx
    group_grid["ymin"] = grid.ymin
    group_grid["ymax"] = grid.ymax
    group_grid["Ny"] = grid.Ny

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
    write_field(group_fdat, dset, field)
    HDF5.attrs(group_fdat[dset])["z"] = z
    HDF5.close(fp)
end


function writeHDF_zdata(plothdf::PlotHDF, z::Float64, pdata::PlotVarData)
    plothdf.iz = plothdf.iz + 1
    iz = plothdf.iz

    fp = HDF5.h5open(plothdf.fname, "r+")

    group_zdat = fp[GROUP_ZDAT]

    data = group_zdat["z"]
    HDF5.set_dims!(data, (iz,))
    data[iz] = FloatGPU(z)

    data = group_zdat["Fzx"]
    # HDF5.set_dims!(data, (iz, grid.Nr))
    # data[iz, :] = pdata.F
    HDF5.set_dims!(data, (length(pdata.F), iz))
    data[:, iz] = pdata.F

    data = group_zdat["Nezx"]
    # HDF5.set_dims!(data, (iz, grid.Nr))
    # data[iz, :] = pdata.rho
    HDF5.set_dims!(data, (length(pdata.rho), iz))
    data[:, iz] = pdata.rho

    data = group_zdat["iSzf"]
    # HDF5.set_dims!(data, (iz, grid.Nw))
    # data[iz, :] = pdata.S
    HDF5.set_dims!(data, (length(pdata.S), iz))
    data[:, iz] = pdata.S

    HDF5.close(fp)
end


function write_field(group, dataset, field::Fields.FieldRT)
    group[dataset] = CuArrays.collect(transpose(real.(field.E)))
    return nothing
end


function write_field(group, dataset, field::Fields.FieldR)
    writeComplexArray(group, dataset, CuArrays.collect(field.E))
    return nothing
end


function write_field(group, dataset, field::Fields.FieldXY)
    writeComplexArray(group, dataset, CuArrays.collect(transpose(field.E)))
    return nothing
end


"""
Adapted from
https://github.com/MagneticParticleImaging/MPIFiles.jl/blob/master/src/Utils.jl
"""
function writeComplexArray(group, dataset,
                           A::AbstractArray{Complex{T}, D}) where {T, D}
    d_type_compound = HDF5.h5t_create(HDF5.H5T_COMPOUND, 2 * sizeof(T))
    HDF5.h5t_insert(d_type_compound, "r", 0 , HDF5.hdf5_type_id(T))
    HDF5.h5t_insert(d_type_compound, "i", sizeof(T) , HDF5.hdf5_type_id(T))

    shape = collect(reverse(size(A)))
    space = HDF5.h5s_create_simple(D, shape, shape)

    dset_compound = HDF5.h5d_create(group, dataset, d_type_compound, space,
                                    HDF5.H5P_DEFAULT, HDF5.H5P_DEFAULT,
                                    HDF5.H5P_DEFAULT)
    HDF5.h5s_close(space)

    HDF5.h5d_write(dset_compound, d_type_compound, HDF5.H5S_ALL, HDF5.H5S_ALL,
                   HDF5.H5P_DEFAULT, A)

    HDF5.h5d_close(dset_compound)
    HDF5.h5t_close(d_type_compound)
    return nothing
end


end

module WritePlots

import CuArrays
import Formatting
import HDF5

import Fields
import Grids
import Units

const FloatGPU = Float32


struct PlotVar{T<:AbstractFloat}
    name :: String
    siunit :: String
    unit :: T
end


# ******************************************************************************
# PlotVarData
# ******************************************************************************
abstract type PlotVarData end


mutable struct PlotVarDataR{T<:AbstractFloat} <: PlotVarData
    plotvars :: Array{PlotVar{T}, 1}
    Imax :: T
    rfil :: T
    P :: T
    I :: Array{FloatGPU, 1}
    Igpu :: CuArrays.CuArray{FloatGPU, 1}
end


mutable struct PlotVarDataT{T<:AbstractFloat} <: PlotVarData
    plotvars :: Array{PlotVar{T}, 1}
    Imax :: T
    rhomax :: T
    duration :: T
    F :: T
end


mutable struct PlotVarDataRT{T<:AbstractFloat} <: PlotVarData
    plotvars :: Array{PlotVar{T}, 1}
    Fmax :: T
    Imax :: T
    rhomax :: T
    De :: T
    rfil :: T
    rpl :: T
    W :: T
    I :: CuArrays.CuArray{FloatGPU, 2}
    F :: Array{FloatGPU, 1}
    rho :: Array{FloatGPU, 1}
    S :: Array{FloatGPU, 1}
end


mutable struct PlotVarDataXY{T<:AbstractFloat} <: PlotVarData
    plotvars :: Array{PlotVar{T}, 1}
    Imax :: T
    ax :: T
    ay :: T
    P :: T
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


function PlotVarData(unit::Units.UnitT, grid::Grids.GridT)
    plotvars = [
        PlotVar("Imax", "W/m2", unit.I),
        PlotVar("Nemax", "1/m3", unit.rho),
        PlotVar("duration", "s", unit.t),
        PlotVar("F", "J/m^2", unit.t * unit.I),
    ]
    return PlotVarDataT(plotvars, 0., 0., 0., 0.)
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


function pdata_update!(
    pdata::PlotVarDataR,
    grid::Grids.GridR,
    field::Fields.FieldR,
)
    pdata.Igpu .= abs2.(field.E)
    pdata.I[:] = CuArrays.collect(pdata.Igpu)

    pdata.Imax = Float64(maximum(pdata.Igpu))
    pdata.rfil = 2. * Grids.radius(grid.r, pdata.I)
    pdata.P = sum(pdata.Igpu .* grid.rdr) * 2. * pi
    return nothing
end


function pdata_update!(
    pdata::PlotVarDataT,
    grid::Grids.GridT,
    field::Fields.FieldT,
)
    pdata.Imax = Fields.peak_intensity(field)
    pdata.rhomax = maximum(field.rho)
    pdata.duration = Fields.pulse_duration(grid, field)
    pdata.F = Fields.peak_fluence(grid, field)
    return nothing
end


function pdata_update!(
    pdata::PlotVarDataRT,
    grid::Grids.GridRT,
    field::Fields.FieldRT,
)
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


function pdata_update!(
    pdata::PlotVarDataXY,
    grid::Grids.GridXY,
    field::Fields.FieldXY,
)
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


function writeDAT(plotdat::PlotDAT, z::AbstractFloat, pdata::PlotVarDataT)
    fp = open(plotdat.fname, "a")
    write(fp, "  $(Formatting.fmt("18.12e", z)) ")   # z
    write(fp, "$(Formatting.fmt("18.12e", pdata.Imax)) ")   # Imax
    write(fp, "$(Formatting.fmt("18.12e", pdata.rhomax)) ")   # Nemax
    write(fp, "$(Formatting.fmt("18.12e", pdata.duration)) ")   # duration
    write(fp, "$(Formatting.fmt("18.12e", pdata.F)) ")   # F
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


mutable struct PlotHDF{T<:AbstractFloat, I<:Int}
    fname :: String
    numplot :: I
    previous_z :: T
    iz :: I
end


function PlotHDF(fname::String, unit::Units.Unit, grid::Grids.Grid)
    fp = HDF5.h5open(fname, "w")
    fp["version"] = PFVERSION
    _write_group_unit(fp, unit)
    _write_group_grid(fp, grid)
    HDF5.g_create(fp, GROUP_FDAT)
    if grid.geometry == "RT"
        _write_group_zdat(fp, grid)
    end
    HDF5.close(fp)

    numplot = 0
    previous_z = -Inf
    iz = 0

    return PlotHDF(fname, numplot, previous_z, iz)
end


function _write_group_unit(fp, unit::Units.UnitR)
    HDF5.g_create(fp, GROUP_UNIT)
    group = fp[GROUP_UNIT]
    group["r"] = unit.r
    group["z"] = unit.z
    group["I"] = unit.I
    return nothing
end


function _write_group_unit(fp, unit::Units.UnitT)
    HDF5.g_create(fp, GROUP_UNIT)
    group = fp[GROUP_UNIT]
    group["z"] = unit.z
    group["t"] = unit.t
    group["I"] = unit.I
    group["Ne"] = unit.rho
    return nothing
end


function _write_group_unit(fp, unit::Units.UnitRT)
    HDF5.g_create(fp, GROUP_UNIT)
    group = fp[GROUP_UNIT]
    group["r"] = unit.r
    group["z"] = unit.z
    group["t"] = unit.t
    group["I"] = unit.I
    group["Ne"] = unit.rho
    return nothing
end


function _write_group_unit(fp, unit::Units.UnitXY)
    HDF5.g_create(fp, GROUP_UNIT)
    group = fp[GROUP_UNIT]
    group["x"] = unit.x
    group["y"] = unit.y
    group["z"] = unit.z
    group["I"] = unit.I
    return nothing
end


function _write_group_grid(fp, grid::Grids.GridR)
    HDF5.g_create(fp, GROUP_GRID)
    group = fp[GROUP_GRID]
    group["geometry"] = grid.geometry
    group["rmax"] = grid.rmax
    group["Nr"] = grid.Nr
    return nothing
end


function _write_group_grid(fp, grid::Grids.GridT)
    HDF5.g_create(fp, GROUP_GRID)
    group = fp[GROUP_GRID]
    group["geometry"] = grid.geometry
    group["tmin"] = grid.tmin
    group["tmax"] = grid.tmax
    group["Nt"] = grid.Nt
    return nothing
end


function _write_group_grid(fp, grid::Grids.GridRT)
    HDF5.g_create(fp, GROUP_GRID)
    group = fp[GROUP_GRID]
    group["geometry"] = grid.geometry
    group["rmax"] = grid.rmax
    group["Nr"] = grid.Nr
    group["tmin"] = grid.tmin
    group["tmax"] = grid.tmax
    group["Nt"] = grid.Nt
    return nothing
end


function _write_group_grid(fp, grid::Grids.GridXY)
    HDF5.g_create(fp, GROUP_GRID)
    group = fp[GROUP_GRID]
    group["geometry"] = grid.geometry
    group["xmin"] = grid.xmin
    group["xmax"] = grid.xmax
    group["Nx"] = grid.Nx
    group["ymin"] = grid.ymin
    group["ymax"] = grid.ymax
    group["Ny"] = grid.Ny
    return nothing
end


function _write_group_zdat(fp, grid::Grids.GridRT)
    HDF5.g_create(fp, GROUP_ZDAT)
    group = fp[GROUP_ZDAT]
    d_create(group, "z", FloatGPU, ((1,), (-1,)))
    # d_create(group, "Fzx", FloatGPU, ((1, grid.Nr), (-1, grid.Nr)))
    d_create(group, "Fzx", FloatGPU, ((grid.Nr, 1), (grid.Nr, -1)))
    # d_create(group, "Nezx", FloatGPU, ((1, grid.Nr), (-1, grid.Nr)))
    d_create(group, "Nezx", FloatGPU, ((grid.Nr, 1), (grid.Nr, -1)))
    # d_create(group, "iSzf", FloatGPU, ((1, grid.Nw), (-1, grid.Nw)))
    d_create(group, "iSzf", FloatGPU, ((grid.Nw, 1), (grid.Nw, -1)))
    return nothing
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


function write_field(group, dataset, field::Fields.FieldR)
    group[dataset] = CuArrays.collect(field.E)
    return nothing
end


function write_field(group, dataset, field::Fields.FieldT)
    group[dataset] = field.E
    return nothing
end


function write_field(group, dataset, field::Fields.FieldRT)
    E = CuArrays.collect(transpose(real.(field.E)))
    shape = size(E)
    typesize = sizeof(eltype(E))
    chunk = guess_chunk(shape, typesize)
    group[dataset, "chunk", chunk, "shuffle", (), "compress", 9] = E
    return nothing
end


function write_field(group, dataset, field::Fields.FieldXY)
    group[dataset] = CuArrays.collect(transpose(field.E))
    return nothing
end


"""
Guess an appropriate chunk layout for a dataset, given its shape and the size of
each element in bytes. Will allocate chunks only as large as MAX_SIZE. Chunks
are generally close to some power-of-2 fraction of each axis, slightly favoring
bigger values for the last index.

Adapted from h5py package:
https://github.com/h5py/h5py/blob/95ff80e0187e8c0c341d097550f09de42d2a4379/h5py/_hl/filters.py#L291
"""
function guess_chunk(shape, typesize)
    CHUNK_BASE = 16 * 1024    # Multiplier by which chunks are adjusted
    CHUNK_MIN = 8 * 1024      # Soft lower limit (8k)
    CHUNK_MAX = 1024 * 1024   # Hard upper limit (1M)

    # For unlimited dimensions we have to guess 1024
    _shape = tuple([x != -1 ? x : 1024 for x in shape]...)

    ndims = length(_shape)
    if ndims == 0
        error("Chunks not allowed for scalar datasets.")
    end

    chunks = [_shape...]
    if ! all(isfinite.(chunks))
        error("Illegal value in chunk tuple")
    end

    # Determine the optimal chunk size in bytes using a PyTables expression.
    # This is kept as a float.
    dset_size = prod(chunks) * typesize
    target_size = CHUNK_BASE * (2^log10(dset_size / (1024 * 1024)))

    if target_size > CHUNK_MAX
        target_size = CHUNK_MAX
    elseif target_size < CHUNK_MIN
        target_size = CHUNK_MIN
    end

    idx = 0
    while true
        # Repeatedly loop over the axes, dividing them by 2.  Stop when:
        # 1a. We're smaller than the target chunk size, OR
        # 1b. We're within 50% of the target chunk size, AND
        #  2. The chunk is smaller than the maximum chunk size

        chunk_bytes = prod(chunks) * typesize

        condition1a = chunk_bytes < target_size
        condition1b = abs(chunk_bytes - target_size) / target_size < 0.5
        condition2 = chunk_bytes < CHUNK_MAX
        if condition1a | condition1b & condition2
            break
        end

        if prod(chunks) == 1
            break   # Element size larger than CHUNK_MAX
        end

        chunks[idx % ndims + 1] = ceil(chunks[idx % ndims + 1] / 2)
        idx = idx + 1
    end

    return tuple(chunks...)
end


function d_create(parent::Union{HDF5.HDF5File, HDF5.HDF5Group},
                  path::String,
                  dtype::Type,
                  dspace_dims::Tuple{Dims,Dims})
     shape = dspace_dims[2]
     typesize = sizeof(dtype)
     chunk = guess_chunk(shape, typesize)
     HDF5.d_create(parent, path, dtype, dspace_dims, "chunk", chunk)
end


end

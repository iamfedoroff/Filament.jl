module WritePlots

import CUDA
import FFTW
import Formatting
import HDF5

import ..Constants: FloatGPU
import ..FieldAnalyzers
import ..Fields
import ..Grids
import ..Units


# ******************************************************************************
# PlotDAT
# ******************************************************************************
struct PlotVar{S<:AbstractString, T<:AbstractFloat}
    name :: S
    siunit :: S
    unit :: T
end


struct PlotDAT{S<:AbstractString}
    fname :: S
end


fmt(x) = Formatting.fmt("18.12e", Float64(x))


function _write_header(fname, plotvars)
    fp = open(fname, "w")

    # write names:
    write(fp, "#")
    for pvar in plotvars
        write(fp, " $(Formatting.fmt("<18", pvar.name))")
    end
    write(fp, "\n")

    # write SI units:
    write(fp, "#")
    for pvar in plotvars
        write(fp, " $(Formatting.fmt("<18", pvar.siunit))")
    end
    write(fp, "\n")

    # write dimensionless units:
    write(fp, "#")
    for pvar in plotvars
        write(fp, " $(Formatting.fmt("<18", pvar.unit))")
    end
    write(fp, "\n")

    close(fp)
    return nothing
end


# ******************************************************************************
# R
# ******************************************************************************
function PlotDAT(fname::String, unit::Units.UnitR)
    plotvars = [
        PlotVar("z", "m", unit.z),
        PlotVar("Imax", "W/m^2", unit.I),
        PlotVar("rfil", "m", unit.r),
        PlotVar("P", "W", unit.r^2 * unit.I),
    ]
    _write_header(fname, plotvars)
    return PlotDAT(fname)
end


function writeDAT(plotdat::PlotDAT, analyzer::FieldAnalyzers.FieldAnalyzerR)
    fp = open(plotdat.fname, "a")
    write(fp, "  ")
    write(fp, "$(fmt(analyzer.z)) ")
    write(fp, "$(fmt(analyzer.Imax)) ")
    write(fp, "$(fmt(analyzer.rfil)) ")
    write(fp, "$(fmt(analyzer.P)) ")
    write(fp, "\n")
    close(fp)
end


# ******************************************************************************
# T
# ******************************************************************************
function PlotDAT(fname::String, unit::Units.UnitT)
    plotvars = [
        PlotVar("z", "m", unit.z),
        PlotVar("Imax", "W/m^2", unit.I),
        PlotVar("Nemax", "1/m^3", unit.rho),
        PlotVar("duration", "s", unit.t),
        PlotVar("F", "J/m^2", unit.t * unit.I),
    ]
    _write_header(fname, plotvars)
    return PlotDAT(fname)
end


function writeDAT(plotdat::PlotDAT, analyzer::FieldAnalyzers.FieldAnalyzerT)
    fp = open(plotdat.fname, "a")
    write(fp, "  ")
    write(fp, "$(fmt(analyzer.z)) ")
    write(fp, "$(fmt(analyzer.Imax)) ")
    write(fp, "$(fmt(analyzer.rhomax)) ")
    write(fp, "$(fmt(analyzer.duration)) ")
    write(fp, "$(fmt(analyzer.F)) ")
    write(fp, "\n")
    close(fp)
end


# ******************************************************************************
# RT
# ******************************************************************************
function PlotDAT(fname::String, unit::Units.UnitRT)
    plotvars = [
        PlotVar("z", "m", unit.z),
        PlotVar("Fmax", "J/m^2", unit.t * unit.I),
        PlotVar("Imax", "W/m^2", unit.I),
        PlotVar("Nemax", "1/m^3", unit.rho),
        PlotVar("De", "1/m", unit.r^2 * unit.rho),
        PlotVar("rfil", "m", unit.r),
        PlotVar("rpl", "m", unit.r),
        PlotVar("tau", "s", unit.t),
        PlotVar("W", "J", unit.r^2 * unit.t * unit.I),
    ]
    _write_header(fname, plotvars)
    return PlotDAT(fname)
end


function writeDAT(plotdat::PlotDAT, analyzer::FieldAnalyzers.FieldAnalyzerRT)
    fp = open(plotdat.fname, "a")
    write(fp, "  ")
    write(fp, "$(fmt(analyzer.z)) ")
    write(fp, "$(fmt(analyzer.Fmax)) ")
    write(fp, "$(fmt(analyzer.Imax)) ")
    write(fp, "$(fmt(analyzer.rhomax)) ")
    write(fp, "$(fmt(analyzer.De)) ")
    write(fp, "$(fmt(analyzer.rfil)) ")
    write(fp, "$(fmt(analyzer.rpl)) ")
    write(fp, "$(fmt(analyzer.tau)) ")
    write(fp, "$(fmt(analyzer.W)) ")
    write(fp, "\n")
    close(fp)
end


# ******************************************************************************
# XY
# ******************************************************************************
function PlotDAT(fname::String, unit::Units.UnitXY)
    plotvars = [
        PlotVar("z", "m", unit.z),
        PlotVar("Imax", "W/m^2", unit.I),
        PlotVar("ax", "m", unit.x),
        PlotVar("ay", "m", unit.y),
        PlotVar("P", "W", unit.x * unit.y * unit.I),
    ]
    _write_header(fname, plotvars)
    return PlotDAT(fname)
end


function writeDAT(plotdat::PlotDAT, analyzer::FieldAnalyzers.FieldAnalyzerXY)
    fp = open(plotdat.fname, "a")
    write(fp, "  ")
    write(fp, "$(fmt(analyzer.z)) ")
    write(fp, "$(fmt(analyzer.Imax)) ")
    write(fp, "$(fmt(analyzer.ax)) ")
    write(fp, "$(fmt(analyzer.ay)) ")
    write(fp, "$(fmt(analyzer.P)) ")
    write(fp, "\n")
    close(fp)
end


# ******************************************************************************
# XYT
# ******************************************************************************
function PlotDAT(fname::String, unit::Units.UnitXYT)
    plotvars = [
        PlotVar("z", "m", unit.z),
        PlotVar("Imax", "W/m^2", unit.I),
        PlotVar("Nemax", "1/m^3", unit.rho),
        PlotVar("Fmax", "J/m^2", unit.t * unit.I),
        PlotVar("ax", "m", unit.x),
        PlotVar("ay", "m", unit.y),
        PlotVar("W", "J", unit.x * unit.y * unit.t * unit.I),
    ]
    _write_header(fname, plotvars)
    return PlotDAT(fname)
end


function writeDAT(plotdat::PlotDAT, analyzer::FieldAnalyzers.FieldAnalyzerXYT)
    fp = open(plotdat.fname, "a")
    write(fp, "  ")
    write(fp, "$(fmt(analyzer.z)) ")
    write(fp, "$(fmt(analyzer.Imax)) ")
    write(fp, "$(fmt(analyzer.rhomax)) ")
    write(fp, "$(fmt(analyzer.Fmax)) ")
    write(fp, "$(fmt(analyzer.ax)) ")
    write(fp, "$(fmt(analyzer.ay)) ")
    write(fp, "$(fmt(analyzer.W)) ")
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


mutable struct PlotHDF{G, T<:AbstractFloat, I<:Int}
    fname :: String
    geometry :: G
    ifield :: I
    dz_field :: T
    zprev_field :: T
    znext_field :: T
    izdata :: I
    dz_zdata :: T
    znext_zdata :: T
    ZDATA :: Bool
end


function PlotHDF(
    fname::String,
    unit::Units.Unit,
    grid::Grids.Grid,
    z::T,
    dz_field::T,
    dz_zdata::T,
) where T<:AbstractFloat
    if (grid isa Grids.GridRT) | (grid isa Grids.GridXY)
        ZDATA = true
    else
        ZDATA = false
    end

    fp = HDF5.h5open(fname, "w")
    fp["version"] = PFVERSION
    _write_group_unit(fp, unit)
    _write_group_grid(fp, grid)
    HDF5.g_create(fp, GROUP_FDAT)
    if ZDATA
        _write_group_zdat(fp, grid)
    end
    HDF5.close(fp)

    ifield = 0
    zprev_field = convert(typeof(z), -Inf)
    znext_field = z

    izdata = 1
    if ZDATA
        znext_zdata = z
    else
        znext_zdata = convert(typeof(z), Inf)
    end

    return PlotHDF(
        fname, typeof(grid),
        ifield, dz_field, zprev_field, znext_field,
        izdata, dz_zdata, znext_zdata, ZDATA,
    )
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


function _write_group_unit(fp, unit::Units.UnitXYT)
    HDF5.g_create(fp, GROUP_UNIT)
    group = fp[GROUP_UNIT]
    group["x"] = unit.x
    group["y"] = unit.y
    group["z"] = unit.z
    group["t"] = unit.t
    group["I"] = unit.I
    group["Ne"] = unit.rho
    return nothing
end


function _write_group_grid(fp, grid::Grids.GridR)
    HDF5.g_create(fp, GROUP_GRID)
    group = fp[GROUP_GRID]
    group["geometry"] = "R"
    group["rmax"] = grid.rmax
    group["Nr"] = grid.Nr
    return nothing
end


function _write_group_grid(fp, grid::Grids.GridRn)
    HDF5.g_create(fp, GROUP_GRID)
    group = fp[GROUP_GRID]
    group["geometry"] = "Rn"
    group["r"] = grid.r
    return nothing
end


function _write_group_grid(fp, grid::Grids.GridT)
    HDF5.g_create(fp, GROUP_GRID)
    group = fp[GROUP_GRID]
    group["geometry"] = "T"
    group["tmin"] = grid.tmin
    group["tmax"] = grid.tmax
    group["Nt"] = grid.Nt
    return nothing
end


function _write_group_grid(fp, grid::Grids.GridRT)
    HDF5.g_create(fp, GROUP_GRID)
    group = fp[GROUP_GRID]
    group["geometry"] = "RT"
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
    group["geometry"] = "XY"
    group["xmin"] = grid.xmin
    group["xmax"] = grid.xmax
    group["Nx"] = grid.Nx
    group["ymin"] = grid.ymin
    group["ymax"] = grid.ymax
    group["Ny"] = grid.Ny
    return nothing
end


function _write_group_grid(fp, grid::Grids.GridXYT)
    HDF5.g_create(fp, GROUP_GRID)
    group = fp[GROUP_GRID]
    group["geometry"] = "XYT"
    group["xmin"] = grid.xmin
    group["xmax"] = grid.xmax
    group["Nx"] = grid.Nx
    group["ymin"] = grid.ymin
    group["ymax"] = grid.ymax
    group["Ny"] = grid.Ny
    group["tmin"] = grid.tmin
    group["tmax"] = grid.tmax
    group["Nt"] = grid.Nt
    return nothing
end


function _write_group_zdat(fp, grid::Grids.GridRT)
    HDF5.g_create(fp, GROUP_ZDAT)
    group = fp[GROUP_ZDAT]

    Nw = length(FFTW.rfftfreq(grid.Nt))

    d_create(group, "z", FloatGPU, ((1,), (-1,)))
    # d_create(group, "Fzx", FloatGPU, ((1, grid.Nr), (-1, grid.Nr)))
    d_create(group, "Fzx", FloatGPU, ((grid.Nr, 1), (grid.Nr, -1)))
    d_create(group, "Ft", FloatGPU, ((grid.Nt, 1), (grid.Nt, -1)))
    # d_create(group, "Nezx", FloatGPU, ((1, grid.Nr), (-1, grid.Nr)))
    d_create(group, "Nezx", FloatGPU, ((grid.Nr, 1), (grid.Nr, -1)))
    # d_create(group, "iSzf", FloatGPU, ((1, Nw), (-1, Nw)))
    d_create(group, "iSzf", FloatGPU, ((Nw, 1), (Nw, -1)))
    return nothing
end


function _write_group_zdat(fp, grid::Grids.GridXY)
    HDF5.g_create(fp, GROUP_ZDAT)
    group = fp[GROUP_ZDAT]
    d_create(group, "z", FloatGPU, ((1,), (-1,)))
    d_create(group, "Izx", FloatGPU, ((1, grid.Nx), (-1, grid.Nx)))
    d_create(group, "Izy", FloatGPU, ((1, grid.Ny), (-1, grid.Ny)))
    return nothing
end


function writeHDF(
    plothdf::PlotHDF,
    field::Fields.Field,
    analyzer::FieldAnalyzers.FieldAnalyzer,
    z::T,
) where T<:AbstractFloat
    if (z != plothdf.zprev_field) & (z >= plothdf.znext_field)
        dset = "$(Formatting.fmt("03d", plothdf.ifield))"
        println(" Writing dataset $(dset)...")

        fp = HDF5.h5open(plothdf.fname, "r+")
        group_fdat = fp[GROUP_FDAT]
        if plothdf.geometry <: Grids.GridR
            write_field_r(group_fdat, dset, field)
        elseif plothdf.geometry <: Grids.GridRn
            write_field_rn(group_fdat, dset, field)
        elseif plothdf.geometry <: Grids.GridT
            write_field_t(group_fdat, dset, field)
        elseif plothdf.geometry <: Grids.GridRT
            write_field_rt(group_fdat, dset, field)
        elseif plothdf.geometry <: Grids.GridXY
            write_field_xy(group_fdat, dset, field)
        elseif plothdf.geometry <: Grids.GridXYT
            write_field_xyt(group_fdat, dset, field)
        else
            error("Wrong grid geometry.")
        end
        HDF5.attrs(group_fdat[dset])["z"] = z
        HDF5.close(fp)

        plothdf.ifield = plothdf.ifield + 1
        plothdf.znext_field = plothdf.znext_field + plothdf.dz_field
    end

    if plothdf.ZDATA
        if z >= plothdf.znext_zdata
            writeHDF_zdata(plothdf, analyzer)
            plothdf.izdata = plothdf.izdata + 1
            plothdf.zprev_field = z
            plothdf.znext_zdata = z + plothdf.dz_zdata
        end
    end
    return nothing
end


function writeHDF_zdata(
    plothdf::PlotHDF, analyzer::FieldAnalyzers.FieldAnalyzerRT,
)
    iz = plothdf.izdata

    fp = HDF5.h5open(plothdf.fname, "r+")

    group_zdat = fp[GROUP_ZDAT]

    data = group_zdat["z"]
    HDF5.set_dims!(data, (iz,))
    data[iz] = analyzer.z

    data = group_zdat["Fzx"]
    # HDF5.set_dims!(data, (iz, grid.Nr))
    # data[iz, :] = analyzer.F
    HDF5.set_dims!(data, (length(analyzer.Fr), iz))
    data[:, iz] = analyzer.Fr

    data = group_zdat["Ft"]
    HDF5.set_dims!(data, (length(analyzer.Ft), iz))
    data[:, iz] = analyzer.Ft

    data = group_zdat["Nezx"]
    # HDF5.set_dims!(data, (iz, grid.Nr))
    # data[iz, :] = analyzer.rho
    HDF5.set_dims!(data, (length(analyzer.rho), iz))
    data[:, iz] = analyzer.rho

    data = group_zdat["iSzf"]
    # HDF5.set_dims!(data, (iz, length(analyzer.S)))
    # data[iz, :] = analyzer.S
    HDF5.set_dims!(data, (length(analyzer.S), iz))
    data[:, iz] = analyzer.S

    HDF5.close(fp)

    return nothing
end


function writeHDF_zdata(
    plothdf::PlotHDF, analyzer::FieldAnalyzers.FieldAnalyzerXY,
)
    iz = plothdf.izdata

    fp = HDF5.h5open(plothdf.fname, "r+")

    group_zdat = fp[GROUP_ZDAT]

    data = group_zdat["z"]
    HDF5.set_dims!(data, (iz,))
    data[iz] = analyzer.z

    Nx, Ny = size(analyzer.I)
    Nx2, Ny2 = Int(Nx / 2), Int(Ny / 2)

    data = group_zdat["Izx"]
    HDF5.set_dims!(data, (iz, Nx))
    data[iz, :] = analyzer.I[:, Ny2]

    data = group_zdat["Izy"]
    HDF5.set_dims!(data, (iz, Ny))
    data[iz, :] = analyzer.I[Nx2, :]

    HDF5.close(fp)

    return nothing
end


function write_field_r(group, dataset, field::Fields.Field)
    group[dataset] = CUDA.collect(field.E)
    return nothing
end


function write_field_rn(group, dataset, field::Fields.Field)
    group[dataset] = field.E
    return nothing
end


function write_field_t(group, dataset, field::Fields.Field)
    group[dataset] = field.E
    return nothing
end


function write_field_rt(group, dataset, field::Fields.Field)
    E = CUDA.collect(transpose(real.(field.E)))
    shape = size(E)
    typesize = sizeof(eltype(E))
    chunk = guess_chunk(shape, typesize)
    group[dataset, "chunk", chunk, "shuffle", (), "compress", 9] = E
    return nothing
end


function write_field_xy(group, dataset, field::Fields.Field)
    group[dataset] = CUDA.collect(transpose(field.E))
    return nothing
end


function write_field_xyt(group, dataset, field::Fields.Field)
    # E = CUDA.collect(real.(field.E))
    # shape = size(E)
    # typesize = sizeof(eltype(E))
    # chunk = guess_chunk(shape, typesize)
    # group[dataset, "chunk", chunk, "shuffle", (), "compress", 9] = E
    group[dataset] = CUDA.collect(real.(field.E))
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

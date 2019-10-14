import Pkg
Pkg.activate(dirname(@__DIR__))

using TimerOutputs
import Formatting
import Dates
import CUDAdrv

push!(LOAD_PATH, joinpath(@__DIR__, "modules"))
import Units
import Grids
import Fields
import Media
import Guards
import Models
import AdaptiveSteps
import Infos
import WritePlots

include("Input.jl")
import .Input


function main()
    # **************************************************************************
    # Prepare units and grid
    # **************************************************************************
    unit = Units.Unit(Input.geometry, Input.p_unit)
    grid = Grids.Grid(Input.geometry, Input.p_grid)

    # **************************************************************************
    # Prepare field
    # **************************************************************************
    z = Input.z / unit.z   # convert initial z to dimensionless units
    field = Fields.Field(unit, grid, Input.p_field)

    # **************************************************************************
    # Prepare medium
    # **************************************************************************
    medium = Media.Medium(Input.permittivity, Input.permeability, Input.n2)

    # **************************************************************************
    # Prepare guards, model, and adaptive z step
    # **************************************************************************
    guard = Guards.Guard(unit, grid, field, medium, Input.p_guard...)
    model = Models.Model(unit, grid, field, medium, guard, Input.p_model...)
    dzadaptive = AdaptiveSteps.AStep(unit, medium, field, Input.p_dzadaptive...)

    # **************************************************************************
    # Prepare output files
    # **************************************************************************
    prefix_dir = dirname(Input.prefix)
    prefix_name = basename(Input.prefix)

    if prefix_dir != ""
        mkpath(prefix_dir)
    end

    file_infos = joinpath(prefix_dir, string(prefix_name, "info.txt"))
    info = Infos.Info(
        file_infos,
        Input.file_input,
        Input.file_initial_condition,
        Input.file_medium,
        unit,
        grid,
        field,
        medium,
    )

    pdata = WritePlots.PlotVarData(unit, grid)
    WritePlots.pdata_update!(pdata, grid, field)

    file_plotdat = joinpath(prefix_dir, string(prefix_name, "plot.dat"))
    plotdat = WritePlots.PlotDAT(file_plotdat, unit, pdata)
    WritePlots.writeDAT(plotdat, z, pdata)

    file_plothdf = joinpath(prefix_dir, string(prefix_name, "plot.h5"))
    plothdf = WritePlots.PlotHDF(file_plothdf, unit, grid)
    WritePlots.writeHDF(plothdf, z, field)
    if grid.geometry == "RT"
        WritePlots.writeHDF_zdata(plothdf, z, pdata)
    end

    # **************************************************************************
    # Main loop
    # **************************************************************************
    stime = Dates.now()

    znext_plothdf = z + Input.dz_plothdf

    if occursin("T", grid.geometry)
        dz_zdata = 0.5 * field.lam0 / unit.z
        znext_zdata = z + dz_zdata
    end

    zfirst = true

    CUDAdrv.synchronize()

    while z < Input.zmax

        if occursin("T", grid.geometry)
            if Input.NONLINEARITY
                dz = dzadaptive(pdata.Imax, pdata.rhomax)
            else
                dz = Input.dz_initial
            end
            println("z=$(Formatting.fmt("18.12e", z))[zu] " *
                    "I=$(Formatting.fmt("18.12e", pdata.Imax))[Iu] " *
                    "rho=$(Formatting.fmt("18.12e", pdata.rhomax))[rhou]")
        else
            if Input.NONLINEARITY
                dz = dzadaptive(pdata.Imax)
            else
                dz = Input.dz_initial
            end
            println("z=$(Formatting.fmt("18.12e", z))[zu] " *
                    "I=$(Formatting.fmt("18.12e", pdata.Imax))[Iu] ")
        end

        dz = min(Input.dz_initial, Input.dz_plothdf, dz)
        z = z + dz

        @timeit "zstep" begin
            Models.zstep(z, dz, grid, field, guard, model)
            CUDAdrv.synchronize()
        end

        @timeit "plots" begin
            # Update plot cache
            @timeit "plot cache" begin
                WritePlots.pdata_update!(pdata, grid, field)
                CUDAdrv.synchronize()
            end

            # Write integral parameters to dat file
            @timeit "writeDAT" begin
                WritePlots.writeDAT(plotdat, z, pdata)
                CUDAdrv.synchronize()
            end

            # Write field to hdf file
            if z >= znext_plothdf
                @timeit "writeHDF" begin
                    WritePlots.writeHDF(plothdf, z, field)
                    znext_plothdf = znext_plothdf + Input.dz_plothdf
                    CUDAdrv.synchronize()
                end
            end

            # Write 1d field data to hdf file
            if occursin("T", grid.geometry)
                if z >= znext_zdata
                    @timeit "writeHDF_zdata" begin
                        WritePlots.writeHDF_zdata(plothdf, z, pdata)
                        znext_zdata = z + dz_zdata
                        CUDAdrv.synchronize()
                    end
                end
            end
        end

        # Exit conditions
        if pdata.Imax > Input.Istop
            message = "Stop (Imax >= Istop): z=$(z)[zu], z=$(z * unit.z)[m]"
            Infos.write_message(info, message)
            break
        end

        # Exclude the first initialization step from timings
        if zfirst
            TimerOutputs.reset_timer!(TimerOutputs.get_defaulttimer())
            zfirst = false
        end

    end

    Infos.write_message(info, TimerOutputs.get_defaulttimer())

    etime = Dates.now()
    ttime = Dates.canonicalize(Dates.CompoundPeriod(etime - stime))
    message = "Start time: $(stime)\n" *
              "End time:   $(etime)\n" *
              "Run time:   $(ttime)"
    Infos.write_message(info, message)
end


main()

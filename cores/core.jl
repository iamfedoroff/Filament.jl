using TimerOutputs
import Formatting
import Dates
import CUDAdrv

push!(LOAD_PATH, joinpath(@__DIR__, "..", "modules"))
import Units
import Grids
import Fields
import Media
import Plasmas
import Infos
import WritePlots
import Models

include("Input.jl")
import .Input


function main()
    # **************************************************************************
    # Prepare units and grid
    # **************************************************************************
    unit = Units.Unit(Input.p_unit...)
    grid = Grids.Grid(Input.p_grid...)

    # **************************************************************************
    # Prepare field
    # **************************************************************************
    z = Input.z / unit.z   # convert initial z to dimensionless units
    field = Fields.Field(unit, grid, Input.lam0, Input.initial_condition)

    # **************************************************************************
    # Prepare medium and plasma
    # **************************************************************************
    medium = Media.Medium(Input.permittivity, Input.permeability, Input.n2)

    if occursin("T", grid.geometry)
        keys = Dict("IONARG" => Input.IONARG, "AVALANCHE" => Input.AVALANCHE)
        plasma = Plasmas.Plasma(unit, grid, field, medium, Input.rho0,
                                Input.nuc, Input.mr, Input.components, keys)
        Plasmas.free_charge(plasma, grid, field)
    end

    # **************************************************************************
    # Prepare output files
    # **************************************************************************
    prefix_dir = dirname(Input.prefix)
    prefix_name = basename(Input.prefix)

    if prefix_dir != ""
        mkpath(prefix_dir)
    end

    file_infos = joinpath(prefix_dir, string(prefix_name, "info.txt"))
    info = Infos.Info(file_infos, Input.file_input,
                      Input.file_initial_condition, Input.file_medium,
                      unit, grid, field, medium)

    pdata = WritePlots.PlotVarData(unit, grid)
    if occursin("T", grid.geometry)
        WritePlots.pdata_update!(pdata, grid, field, plasma)
    else
        WritePlots.pdata_update!(pdata, grid, field)
    end

    file_plotdat = joinpath(prefix_dir, string(prefix_name, "plot.dat"))
    plotdat = WritePlots.PlotDAT(file_plotdat, unit, pdata)
    WritePlots.writeDAT(plotdat, z, pdata)

    file_plothdf = joinpath(prefix_dir, string(prefix_name, "plot.h5"))
    plothdf = WritePlots.PlotHDF(file_plothdf, unit, grid)
    WritePlots.writeHDF(plothdf, z, field)
    if occursin("T", grid.geometry)
        WritePlots.writeHDF_zdata(plothdf, z, pdata)
    end

    # **************************************************************************
    # Prepare model
    # **************************************************************************
    if occursin("T", grid.geometry)
        keys = Dict(
            "KPARAXIAL" => Input.KPARAXIAL, "QPARAXIAL" => Input.QPARAXIAL,
            "rguard" => Input.rguard, "tguard" => Input.tguard,
            "kguard" => Input.kguard, "wguard" => Input.wguard,
            "RKORDER" => Input.RKORDER)
        model = Models.Model(unit, grid, field, medium, plasma, keys,
                             Input.responses)
    else
        keys = Dict(
            "KPARAXIAL" => Input.KPARAXIAL, "QPARAXIAL" => Input.QPARAXIAL,
            "rguard" => Input.rguard, "kguard" => Input.kguard,
            "RKORDER" => Input.RKORDER)
        model = Models.Model(unit, grid, field, medium, keys, Input.responses)
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
            println("z=$(Formatting.fmt("18.12e", z))[zu] " *
                    "I=$(Formatting.fmt("18.12e", pdata.Imax))[Iu] " *
                    "rho=$(Formatting.fmt("18.12e", pdata.rhomax))[rhou]")
        else
            println("z=$(Formatting.fmt("18.12e", z))[zu] " *
                    "I=$(Formatting.fmt("18.12e", pdata.Imax))[Iu] ")
        end

        # Adaptive z step
        if occursin("T", grid.geometry)
            dz = Models.adaptive_dz(model, Input.dzAdaptLevel, pdata.Imax,
                                    pdata.rhomax)
        else
            dz = Models.adaptive_dz(model, Input.dzAdaptLevel, pdata.Imax)
        end
        dz = min(Input.dz_initial, Input.dz_plothdf, dz)
        z = z + dz

        @timeit "zstep" begin
            if occursin("T", grid.geometry)
                Models.zstep(z, dz, grid, field, plasma, model)
            else
                Models.zstep(z, dz, grid, field, model)
            end
        end

        @timeit "plots" begin
            # Update plot cache
            @timeit "plot cache" begin
                if occursin("T", grid.geometry)
                    WritePlots.pdata_update!(pdata, grid, field, plasma)
                else
                    WritePlots.pdata_update!(pdata, grid, field)
                end
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

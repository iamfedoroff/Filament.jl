using TimerOutputs
import Formatting
import Dates

push!(LOAD_PATH, joinpath(@__DIR__, "..", "modules"))
import Units
import Grids
import Fields
import Media
import Plasmas
import Infos
import WritePlots
import Models

const timer = TimerOutputs.TimerOutput()


module Input
    # Modules and variables available in input files:
    import FFTW

    import PyCall
    numpy_fft = PyCall.pyimport("numpy.fft")
    scipy_constants = PyCall.pyimport("scipy.constants")
    C0 = scipy_constants.c   # speed of light in vacuum

    # Read input file and change current working directory:
    file_input = abspath(ARGS[1])
    include(file_input)
    cd(dirname(file_input))

    # Read initial condition file:
    file_initial_condition = abspath(file_initial_condition)
    include(file_initial_condition)

    # Read medium file:
    file_medium = abspath(file_medium)
    include(file_medium)
end


import .Input


function main()
    # **************************************************************************
    # Prepare units and grid
    # **************************************************************************
    unit = Units.Unit(Input.ru, Input.zu, Input.tu, Input.Iu, Input.rhou)
    grid = Grids.Grid(Input.rmax, Input.Nr, Input.tmin, Input.tmax, Input.Nt)

    # **************************************************************************
    # Read the initial condition file and prepare field
    # **************************************************************************
    z = Input.z / unit.z   # convert initial z to dimensionless units
    field = Fields.Field(unit, grid, Input.lam0, Input.initial_condition)

    # **************************************************************************
    # Read the medium file and prepare medium and plasma
    # **************************************************************************
    medium = Media.Medium(Input.permittivity, Input.permeability, Input.n2,
                          Input.raman_response, Input.graman)

    keys = Dict("IONARG" => Input.IONARG, "AVALANCHE" => Input.AVALANCHE)
    plasma = Plasmas.Plasma(unit, grid, field, medium, Input.rho0, Input.nuc,
                            Input.mr, Input.components, keys)
    Plasmas.free_charge(plasma, grid, field)

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
                      unit, grid, field, medium, plasma)

    pcache = WritePlots.PlotCache(grid)
    WritePlots.plotcache_update!(pcache, grid, field, plasma)

    file_plotdat = joinpath(prefix_dir, string(prefix_name, "plot.dat"))
    plotdat = WritePlots.PlotDAT(file_plotdat, unit)
    WritePlots.writeDAT(plotdat, z, pcache)

    file_plothdf = joinpath(prefix_dir, string(prefix_name, "plot.h5"))
    plothdf = WritePlots.PlotHDF(file_plothdf, unit, grid)
    WritePlots.writeHDF(plothdf, z, field)
    WritePlots.writeHDF_zdata(plothdf, z, pcache)

    # **************************************************************************
    # Prepare model
    # **************************************************************************
    keys = Dict(
        "KPARAXIAL" => Input.KPARAXIAL, "QPARAXIAL" => Input.QPARAXIAL,
        "KERR" => Input.KERR, "THG" => Input.THG, "RAMAN" => Input.RAMAN,
        "RTHG" => Input.RTHG, "PLASMA" => Input.PLASMA,
        "ILOSSES" => Input.ILOSSES, "IONARG" => Input.IONARG,
        "rguard_width" => Input.rguard_width,
        "tguard_width" => Input.tguard_width, "kguard" => Input.kguard,
        "wguard" => Input.wguard, "RKORDER" => Input.RKORDER)
    model = Models.Model(unit, grid, field, medium, plasma, keys)

    # **************************************************************************
    # Main loop
    # **************************************************************************
    stime = Dates.now()

    znext_plothdf = z + Input.dz_plothdf

    dz_zdata = 0.5 * field.lam0
    znext_zdata = z + dz_zdata

    @timeit timer "main loop" while z < Input.zmax

        println("z=$(Formatting.fmt("18.12e", z))[zu] " *
                "I=$(Formatting.fmt("18.12e", pcache.Imax))[Iu] " *
                "rho=$(Formatting.fmt("18.12e", pcache.rhomax))[rhou]")

        # Adaptive z step
        dz = Models.adaptive_dz(model, Input.dzAdaptLevel, pcache.Imax,
                                pcache.rhomax)
        dz = min(Input.dz_initial, Input.dz_plothdf, dz)
        z = z + dz

        @timeit timer "zstep" begin
            Models.zstep(dz, grid, field, plasma, model, timer)
        end

        @timeit timer "plots" begin
            # Update plot cache
            @timeit timer "plot cache" begin
                WritePlots.plotcache_update!(pcache, grid, field, plasma)
            end

            # Write integral parameters to dat file
            @timeit timer "writeDAT" begin
                WritePlots.writeDAT(plotdat, z, pcache)
            end

            # Write field to hdf file
            if z >= znext_plothdf
                @timeit timer "writeHDF" begin
                    WritePlots.writeHDF(plothdf, z, field)
                    znext_plothdf = znext_plothdf + Input.dz_plothdf
                end
            end

            # Write 1d field data to hdf file
            if z >= znext_zdata
                @timeit timer "writeHDF_zdata" begin
                    WritePlots.writeHDF_zdata(plothdf, z, pcache)
                    znext_zdata = z + dz_zdata
                end
            end
        end

        # Exit conditions
        if pcache.Imax > Input.Istop
            message = "Stop (Imax >= Istop): z=$(z)[zu], z=$(z * unit.z)[m]"
            Infos.write_message(info, message)
            break
        end

    end

    Infos.write_message(info, timer)

    etime = Dates.now()
    ttime = Dates.canonicalize(Dates.CompoundPeriod(etime - stime))
    message = "Start time: $(stime)\n" *
              "End time:   $(etime)\n" *
              "Run time:   $(ttime)"
    Infos.write_message(info, message)
end


main()

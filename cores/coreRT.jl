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


module Input
    # Modules and variables available in input files:
    using PyCall
    @pyimport numpy.fft as npfft
    @pyimport scipy.constants as sc
    C0 = sc.c   # speed of light in vacuum

    # Read input file and change current working directory:
    file_input = abspath(ARGS[1])
    include(file_input)
    cd(dirname(file_input))

    # Read initial condition file:
    include(abspath(file_initial_condition))

    # Read medium file:
    include(abspath(file_medium))
end

import Input


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

    file_plotdat = joinpath(prefix_dir, string(prefix_name, "plot.dat"))
    plotdat = WritePlots.PlotDAT(file_plotdat, unit)
    WritePlots.writeDAT(plotdat, z, field)

    file_plothdf = joinpath(prefix_dir, string(prefix_name, "plot.h5"))
    plothdf = WritePlots.PlotHDF(file_plothdf, unit, grid)
    WritePlots.writeHDF(plothdf, z, field)
    WritePlots.writeHDF_zdata(plothdf, z, field)

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
        "wguard" => Input.wguard, "FFTWFLAG" => Input.FFTWFLAG)
    model = Models.Model(unit, grid, field, medium, plasma, keys)

    # **************************************************************************
    # Main loop
    # **************************************************************************
    stime = now()

    znext_plothdf = z + Input.dz_plothdf

    dz_zdata = 0.5 * field.lam0
    znext_zdata = z + dz_zdata

    @time while z < Input.zmax
        Imax = Fields.peak_intensity(field)
        rhomax = Fields.peak_plasma_density(field)

        # Adaptive z step
        dz = Models.adaptive_dz(model, Input.dzAdaptLevel, Imax, rhomax)
        dz = min(Input.dz_initial, Input.dz_plothdf, dz)
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
            znext_plothdf = znext_plothdf + Input.dz_plothdf
        end

        # Write 1d field data to hdf file
        if z >= znext_zdata
            WritePlots.writeHDF_zdata(plothdf, z, field)
            znext_zdata = z + dz_zdata
        end

        # Exit conditions
        if Imax > Input.Istop
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
end


main()

module Filament

# Global packages --------------------------------------------------------------
import CUDA
import Dates
import Formatting
using TimerOutputs

# Local package-like modules ---------------------------------------------------
include(joinpath("modules", "AnalyticSignals.jl"))
import .AnalyticSignals

include(joinpath("modules", "Equations.jl"))
import .Equations

include(joinpath("modules", "TabulatedFunctions.jl"))
import .TabulatedFunctions

# Local modules ----------------------------------------------------------------
include(joinpath("modules", "Constants.jl"))
import .Constants

include(joinpath("modules", "Units.jl"))
import .Units

include(joinpath("modules", "Grids.jl"))
import .Grids

include(joinpath("modules", "Fields.jl"))
import .Fields

include(joinpath("modules", "Media.jl"))
import .Media

include(joinpath("modules", "Guards.jl"))
import .Guards

include(joinpath("modules", "Models.jl"))
import .Models

include(joinpath("modules", "FieldAnalyzers.jl"))
import .FieldAnalyzers

include(joinpath("modules", "AdaptiveSteps.jl"))
import .AdaptiveSteps

include(joinpath("modules", "Infos.jl"))
import .Infos

include(joinpath("modules", "WritePlots.jl"))
import .WritePlots

# Input ------------------------------------------------------------------------
include("Input.jl")
import .Input


prepare = Input.prepare


function run(input)
    prefix = input["prefix"]
    geometry = input["geometry"]
    z = input["z"]
    zmax = input["zmax"]
    lam0 = input["lam0"]
    dz_plothdf = input["dz_plothdf"]
    Istop = input["Istop"]
    p_unit = input["p_unit"]
    p_grid = input["p_grid"]
    p_field = input["p_field"]
    p_medium = input["p_medium"]
    p_guard = input["p_guard"]
    p_model = input["p_model"]
    p_dzadaptive = input["p_dzadaptive"]
    p_info = input["p_info"]

    # Prepare data structures --------------------------------------------------
    unit = Units.Unit(geometry, p_unit)
    grid = Grids.Grid(geometry, p_grid)
    field = Fields.Field(unit, grid, p_field)
    medium = Media.Medium(p_medium...)
    guard = Guards.Guard(unit, grid, field, medium, p_guard...)
    model = Models.Model(unit, grid, field, medium, guard, p_model...)
    dzadaptive = AdaptiveSteps.AStep(unit, medium, field, p_dzadaptive...)
    analyzer = FieldAnalyzers.FieldAnalyzer(grid, field, z)

    FieldAnalyzers.analyze!(analyzer, grid, field, z)

    # Prepare output files -----------------------------------------------------
    prefix_dir = dirname(prefix)
    prefix_name = basename(prefix)

    if prefix_dir != ""
        mkpath(prefix_dir)
    end

    file_info = joinpath(prefix_dir, string(prefix_name, "info.txt"))
    info = Infos.Info(file_info, unit, grid, field, medium, analyzer, p_info...)

    file_plotdat = joinpath(prefix_dir, string(prefix_name, "plot.dat"))
    plotdat = WritePlots.PlotDAT(file_plotdat, unit)
    WritePlots.writeDAT(plotdat, analyzer)

    dz_zdata = convert(typeof(lam0), lam0 / 2 / unit.z)
    file_plothdf = joinpath(prefix_dir, string(prefix_name, "plot.h5"))
    plothdf = WritePlots.PlotHDF(
        file_plothdf, unit, grid, z, dz_plothdf, dz_zdata,
    )
    WritePlots.writeHDF(plothdf, field, analyzer, z)

    # Main loop ----------------------------------------------------------------
    main_loop(
        z, zmax, unit, grid, field, guard, model, dzadaptive, analyzer, info,
        plotdat, plothdf, Istop,
    )
    return nothing
end


function main_loop(
    z, zmax, unit, grid, field, guard, model, dzadaptive, analyzer, info,
    plotdat, plothdf, Istop,
)
    fmt(x) = Formatting.fmt("18.12e", Float64(x))   # output print format

    stime = Dates.now()

    zfirst = true

    CUDA.synchronize()

    while z < zmax

        if isa(grid, Grids.GridT) | isa(grid, Grids.GridRT)
            println("z=$(fmt(z))[zu] I=$(fmt(analyzer.Imax))[Iu]" *
                    " rho=$(fmt(analyzer.rhomax))[rhou]")
        else
            println("z=$(fmt(z))[zu] I=$(fmt(analyzer.Imax))[Iu]")
        end

        dz = dzadaptive(analyzer)
        z = z + dz

        @timeit "zstep" begin
            Models.zstep(z, dz, grid, field, guard, model)
            CUDA.synchronize()
        end

        @timeit "plots" begin
            # Update plot cache
            @timeit "field analyzer" begin
                FieldAnalyzers.analyze!(analyzer, grid, field, z)
                CUDA.synchronize()
            end

            # Write integral parameters to dat file
            @timeit "writeDAT" begin
                WritePlots.writeDAT(plotdat, analyzer)
                CUDA.synchronize()
            end

            # Write field to hdf file
            @timeit "writeHDF" begin
                WritePlots.writeHDF(plothdf, field, analyzer, z)
                CUDA.synchronize()
            end
        end

        if analyzer.Imax > Istop
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

    return nothing
end


end

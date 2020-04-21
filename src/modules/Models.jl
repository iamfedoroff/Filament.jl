module Models

import CUDAdrv
using TimerOutputs

import Fields
import FourierTransforms
import Grids
import Guards
import LinearPropagators
import Media
import NonlinearPropagators
import PlasmaEquations
import Units


struct Model{TLP, TNP, TPE, B<:Bool}
    LP :: TLP
    NP :: TNP
    PE :: TPE
    NONLINEARITY :: B
    PLASMA :: B
end


function Model(
    unit::Units.Unit,
    grid::Grids.Grid,
    field::Fields.Field,
    medium::Media.Medium,
    guard::Guards.Guard,
    responses_list::AbstractArray,
    plasma_equation::Dict,
    keys::NamedTuple,
)
    NONLINEARITY = keys.NONLINEARITY
    PLASMA = keys.PLASMA

    LP = LinearPropagators.LinearPropagator(
        unit, grid, medium, field, guard, keys.KPARAXIAL,
    )

    if NONLINEARITY
        NP = NonlinearPropagators.NonlinearPropagator(
            unit, grid, medium, field, guard, responses_list, keys,
        )
    else
        NP = nothing
    end

    if PLASMA
        w0 = field.w0
        n0 = Media.refractive_index(medium, w0)
        PE = PlasmaEquations.PlasmaEquation(unit, n0, w0, plasma_equation)
        PlasmaEquations.solve!(PE, field.rho, field.kdrho, grid.t, field.E)
    else
        PE = nothing
    end

    return Model(LP, NP, PE, NONLINEARITY, PLASMA)
end


function zstep(
    z::T,
    dz::T,
    grid::Grids.Grid,
    field::Fields.Field,
    guard::Guards.Guard,
    model::Model,
) where T<:AbstractFloat
    # Calculate plasma density:
    if model.PLASMA
        @timeit "plasma" begin
            PlasmaEquations.solve!(
                model.PE, field.rho, field.kdrho, grid.t, field.E,
            )
            CUDAdrv.synchronize()
        end
    end

    if isa(grid, Grids.GridT) | isa(grid, Grids.GridRT)
        @timeit "field -> spectr" begin
            FourierTransforms.fft!(field.E, field.FT)
            CUDAdrv.synchronize()
        end
    end

    if model.NONLINEARITY
        @timeit "nonlinearity" begin
           NonlinearPropagators.propagate!(field.E, model.NP, z, dz)
           CUDAdrv.synchronize()
       end
    end

    @timeit "linear" begin
        LinearPropagators.propagate!(field.E, model.LP, dz)
        CUDAdrv.synchronize()
    end

    if isa(grid, Grids.GridT) | isa(grid, Grids.GridRT)
        @timeit "spectr -> field" begin
            FourierTransforms.ifft!(field.E, field.FT)
            CUDAdrv.synchronize()
        end
    end

    @timeit "field filter" begin
        Guards.apply_field_filter!(field.E, guard)
        CUDAdrv.synchronize()
    end
    return nothing
end


end

module Models

import CUDAdrv
using TimerOutputs

import Constants: FloatGPU
import Fields
import Fourier
import Grids
import Guards
import LinearPropagators
import Media
import NonlinearPropagators
import PlasmaEquations
import Units


abstract type Model end


struct ModelR <: Model
    LP :: LinearPropagators.LinearPropagator
    NP :: Union{NonlinearPropagators.NonlinearPropagator, Nothing}
    keys :: NamedTuple
end


struct ModelT <: Model
    LP :: LinearPropagators.LinearPropagator
    NP :: Union{NonlinearPropagators.NonlinearPropagator, Nothing}
    PE :: Union{PlasmaEquations.PlasmaEquation, Nothing}
    keys :: NamedTuple
end


struct ModelRT <: Model
    LP :: LinearPropagators.LinearPropagator
    NP :: Union{NonlinearPropagators.NonlinearPropagator, Nothing}
    PE :: Union{PlasmaEquations.PlasmaEquation, Nothing}
    keys :: NamedTuple
end


struct ModelXY <: Model
    LP :: LinearPropagators.LinearPropagator
    NP :: Union{NonlinearPropagators.NonlinearPropagator, Nothing}
    keys :: NamedTuple
end


function Model(
    unit::Units.UnitR,
    grid::Grids.GridR,
    field::Fields.FieldR,
    medium::Media.Medium,
    guard::Guards.GuardR,
    responses_list::AbstractArray,
    keys::NamedTuple,
)
    LP = LinearPropagators.LinearPropagator(
        unit, grid, medium, field, guard, keys,
    )

    if keys.NONLINEARITY
        NP = NonlinearPropagators.NonlinearPropagator(
            unit, grid, medium, field, guard, responses_list, keys,
        )
    else
        NP = nothing
    end

    return ModelR(LP, NP, keys)
end


function Model(
    unit::Units.UnitT,
    grid::Grids.GridT,
    field::Fields.FieldT,
    medium::Media.Medium,
    guard::Guards.GuardT,
    responses_list::AbstractArray,
    plasma_equation::Dict,
    keys::NamedTuple,
)
    LP = LinearPropagators.LinearPropagator(
        unit, grid, medium, field, guard, keys,
    )

    if keys.NONLINEARITY
        NP = NonlinearPropagators.NonlinearPropagator(
            unit, grid, medium, field, guard, responses_list, keys,
        )
    else
        NP = nothing
    end

    if keys.PLASMA
        w0 = field.w0
        n0 = Media.refractive_index(medium, w0)
        PE = PlasmaEquations.PlasmaEquation(unit, n0, w0, plasma_equation)
        PlasmaEquations.solve!(PE, field.rho, field.kdrho, grid.t, field.E)
    else
        PE = nothing
    end

    return ModelT(LP, NP, PE, keys)
end


function Model(
    unit::Units.UnitRT,
    grid::Grids.GridRT,
    field::Fields.FieldRT,
    medium::Media.Medium,
    guard::Guards.GuardRT,
    responses_list::AbstractArray,
    plasma_equation::Dict,
    keys::NamedTuple,
)
    LP = LinearPropagators.LinearPropagator(
        unit, grid, medium, field, guard, keys,
    )

    if keys.NONLINEARITY
        NP = NonlinearPropagators.NonlinearPropagator(
            unit, grid, medium, field, guard, responses_list, keys,
        )
    else
        NP = nothing
    end

    if keys.PLASMA
        w0 = field.w0
        n0 = Media.refractive_index(medium, w0)
        n0 = convert(eltype(w0), real(n0))   # FIXME Should be removed when Media will be parametrized
        PE = PlasmaEquations.PlasmaEquation(unit, n0, w0, plasma_equation)
        PlasmaEquations.solve!(PE, field.rho, field.kdrho, grid.t, field.E)
    else
        PE = nothing
    end

    return ModelRT(LP, NP, PE, keys)
end


function Model(
    unit::Units.UnitXY,
    grid::Grids.GridXY,
    field::Fields.FieldXY,
    medium::Media.Medium,
    guard::Guards.GuardXY,
    responses_list::AbstractArray,
    keys::NamedTuple,
)
    LP = LinearPropagators.LinearPropagator(
        unit, grid, medium, field, guard, keys,
    )

    if keys.NONLINEARITY
        NP = NonlinearPropagators.NonlinearPropagator(
            unit, grid, medium, field, guard, responses_list, keys,
        )
    else
        NP = nothing
    end

    return ModelXY(LP, NP, keys)
end


function zstep(
    z::T,
    dz::T,
    grid::Union{Grids.GridR, Grids.GridXY},
    field::Union{Fields.FieldR, Fields.FieldXY},
    guard::Guards.Guard,
    model::Model,
) where T
    z = convert(FloatGPU, z)
    dz = convert(FloatGPU, dz)

    if model.keys.NONLINEARITY
        @timeit "nonlinearity" begin
           NonlinearPropagators.propagate!(field.E, model.NP, z, dz)
           CUDAdrv.synchronize()
       end
    end

    @timeit "linear" begin
        LinearPropagators.propagate!(field.E, model.LP, dz)
        CUDAdrv.synchronize()
    end

    @timeit "field filter" begin
        Guards.apply_field_filter!(field.E, guard)
        CUDAdrv.synchronize()
    end

    return nothing
end


function zstep(
    z::T,
    dz::T,
    grid::Grids.GridT,
    field::Fields.FieldT,
    guard::Guards.GuardT,
    model::ModelT,
) where T
    # Calculate plasma density:
    if model.keys.PLASMA
        @timeit "plasma" begin
            PlasmaEquations.solve!(
                model.PE, field.rho, field.kdrho, grid.t, field.E,
            )
        end
    end

    # Field -> temporal spectrum:
    @timeit "field -> spectr" begin
        Fourier.rfft!(field.S, field.FT, field.E)
    end

    if model.keys.NONLINEARITY
        @timeit "nonlinearity" begin
           NonlinearPropagators.propagate!(field.S, model.NP, z, dz)
       end
    end

    @timeit "linear" begin
        LinearPropagators.propagate!(field.S, model.LP, dz)
    end

    # Temporal spectrum -> field:
    @timeit "spectr -> field" begin
        Fourier.hilbert!(field.E, field.FT, field.S)   # spectrum real to signal analytic
    end

    @timeit "field filter" begin
        Guards.apply_field_filter!(field.E, guard)
    end

    return nothing
end


function zstep(
    z::T,
    dz::T,
    grid::Grids.GridRT,
    field::Fields.FieldRT,
    guard::Guards.GuardRT,
    model::ModelRT,
) where T
    z = convert(FloatGPU, z)
    dz = convert(FloatGPU, dz)

    # Calculate plasma density:
    if model.keys.PLASMA
        @timeit "plasma" begin
            PlasmaEquations.solve!(
                model.PE, field.rho, field.kdrho, grid.t, field.E,
            )
            CUDAdrv.synchronize()
        end
    end

    # Field -> temporal spectrum:
    @timeit "field -> spectr" begin
        Fourier.rfft!(field.S, field.FT, field.E)
        CUDAdrv.synchronize()
    end

    if model.keys.NONLINEARITY
        @timeit "nonlinearity" begin
           NonlinearPropagators.propagate!(field.S, model.NP, z, dz)
           CUDAdrv.synchronize()
       end
    end

    @timeit "linear" begin
        LinearPropagators.propagate!(field.S, model.LP, dz)
        CUDAdrv.synchronize()
    end

    # Temporal spectrum -> field:
    @timeit "spectr -> field" begin
        Fourier.hilbert!(field.E, field.FT, field.S)   # spectrum real to signal analytic
        CUDAdrv.synchronize()
    end

    @timeit "field filter" begin
        Guards.apply_field_filter!(field.E, guard)
        CUDAdrv.synchronize()
    end

    return nothing
end


end

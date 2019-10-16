module Models

import CUDAdrv
using TimerOutputs

import Fields
import Fourier
import Grids
import Guards
import LinearPropagators
import Media
import NonlinearPropagators
import PlasmaEquations
import Units

const FloatGPU = Float32


abstract type Model end


struct ModelR <: Model
    LP :: LinearPropagators.LinearPropagator
    NP :: NonlinearPropagators.NonlinearPropagator
    keys :: NamedTuple
end


struct ModelT <: Model
    LP :: LinearPropagators.LinearPropagator
    NP :: NonlinearPropagators.NonlinearPropagator
    PE :: PlasmaEquations.PlasmaEquation
    keys :: NamedTuple
end


struct ModelRT <: Model
    LP :: LinearPropagators.LinearPropagator
    NP :: NonlinearPropagators.NonlinearPropagator
    PE :: PlasmaEquations.PlasmaEquation
    keys :: NamedTuple
end


struct ModelXY <: Model
    LP :: LinearPropagators.LinearPropagator
    NP :: NonlinearPropagators.NonlinearPropagator
    keys :: NamedTuple
end


function Model(
    unit::Units.UnitR,
    grid::Grids.GridR,
    field::Fields.FieldR,
    medium::Media.Medium,
    guard::Guards.GuardR,
    responses_list,
    keys::NamedTuple,
)
    LP = LinearPropagators.LinearPropagator(
        unit, grid, medium, field, guard, keys,
    )

    NP = NonlinearPropagators.NonlinearPropagator(
        unit, grid, medium, field, guard, responses_list, keys,
    )

    return ModelR(LP, NP, keys)
end


function Model(
    unit::Units.UnitT,
    grid::Grids.GridT,
    field::Fields.FieldT,
    medium::Media.Medium,
    guard::Guards.GuardT,
    responses_list,
    plasma_equation::Dict,
    keys::NamedTuple,
)
    LP = LinearPropagators.LinearPropagator(
        unit, grid, medium, field, guard, keys,
    )

    NP = NonlinearPropagators.NonlinearPropagator(
        unit, grid, medium, field, guard, responses_list, keys,
    )

    # Plasma equation:
    n0 = Media.refractive_index(medium, field.w0)
    PE = PlasmaEquations.PlasmaEquation(unit, n0, field.w0, plasma_equation)
    if keys.NONLINEARITY
        PE.solve!(field.rho, field.Kdrho, grid.t, field.E)
    end

    return ModelT(LP, NP, PE, keys)
end


function Model(
    unit::Units.UnitRT,
    grid::Grids.GridRT,
    field::Fields.FieldRT,
    medium::Media.Medium,
    guard::Guards.GuardRT,
    responses_list,
    plasma_equation::Dict,
    keys::NamedTuple,
)
    LP = LinearPropagators.LinearPropagator(
        unit, grid, medium, field, guard, keys,
    )

    NP = NonlinearPropagators.NonlinearPropagator(
        unit, grid, medium, field, guard, responses_list, keys,
    )

    # Plasma equation:
    n0 = Media.refractive_index(medium, field.w0)
    PE = PlasmaEquations.PlasmaEquation(unit, n0, field.w0, plasma_equation)
    if keys.NONLINEARITY
        t = range(convert(FloatGPU, grid.tmin),
                  convert(FloatGPU, grid.tmax), length=grid.Nt)
        PE.solve!(field.rho, field.Kdrho, t, field.E)
    end

    return ModelRT(LP, NP, PE, keys)
end


function Model(
    unit::Units.UnitXY,
    grid::Grids.GridXY,
    field::Fields.FieldXY,
    medium::Media.Medium,
    guard::Guards.GuardXY,
    responses_list,
    keys::NamedTuple,
)
    LP = LinearPropagators.LinearPropagator(
        unit, grid, medium, field, guard, keys,
    )

    NP = NonlinearPropagators.NonlinearPropagator(
        unit, grid, medium, field, guard, responses_list, keys,
    )

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
        Guards.apply_field_filter!(guard, field.E)
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
    if model.keys.NONLINEARITY
        @timeit "plasma" begin
            model.PE.solve!(field.rho, field.Kdrho, grid.t, field.E)
        end
    end

    # Field -> temporal spectrum:
    @timeit "field -> spectr" begin
        Fourier.rfft!(grid.FT, field.E, field.S)
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
        Fourier.hilbert!(grid.FT, field.S, field.E)   # spectrum real to signal analytic
    end

    @timeit "field filter" begin
        Guards.apply_field_filter!(guard, field.E)
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
    if model.keys.NONLINEARITY
        @timeit "plasma" begin
            t = range(convert(FloatGPU, grid.tmin),
                      convert(FloatGPU, grid.tmax), length=grid.Nt)
            model.PE.solve!(field.rho, field.Kdrho, t, field.E)
            CUDAdrv.synchronize()
        end
    end

    # Field -> temporal spectrum:
    @timeit "field -> spectr" begin
        Fourier.rfft2!(grid.FT, field.E, field.S)
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
        Fourier.hilbert2!(grid.FT, field.S, field.E)   # spectrum real to signal analytic
        CUDAdrv.synchronize()
    end

    @timeit "field filter" begin
        Guards.apply_field_filter!(guard, field.E)
        CUDAdrv.synchronize()
    end

    return nothing
end


end

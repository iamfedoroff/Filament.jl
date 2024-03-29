module Models

import AnalyticSignals
import CUDA
import FFTW
import HankelTransforms
import ODEIntegrators
import StaticArrays
using TimerOutputs

import ..Constants: FloatGPU, MU0
import ..Fields
import ..Grids
import ..Guards
import ..Media
import ..Units

include(joinpath("linear_propagators", "spectral.jl"))
include(joinpath("linear_propagators", "tdma.jl"))
include("nonlinear_propagators.jl")
include("plasma_equations.jl")


struct Model{TLP, TNP, TPE, B<:Bool}
    LP :: TLP
    NP :: TNP
    pe :: TPE
    NONLINEARITY :: B
    PLASMA :: B
end


function Model(
    unit::Units.Unit,
    grid::Grids.Grid,
    field::Fields.Field,
    medium::Media.Medium,
    guard::Guards.Guard,
    responses::AbstractArray,
    plasma_equation::Dict,
    keys::NamedTuple,
)
    NONLINEARITY = keys.NONLINEARITY
    PLASMA = keys.PLASMA

    LP = LinearPropagator(unit, grid, medium, field, guard, keys.KPARAXIAL)

    if NONLINEARITY
        NP = NonlinearPropagator(
            unit, grid, medium, field, guard, responses, keys.QPARAXIAL,
            keys.ALG,
        )
    else
        NP = nothing
    end

    if PLASMA
        w0 = field.w0
        n0 = Media.refractive_index(medium, w0)
        Eu = Units.E(unit, real(n0))
        units = (unit.t, Eu, unit.I, unit.rho)

        init = plasma_equation["init"]
        pe = init(grid.t, field.E, w0, units, plasma_equation)

        solve!(field.rho, field.kdrho, field.E, grid.t, pe)
    else
        pe = nothing
    end

    return Model(LP, NP, pe, NONLINEARITY, PLASMA)
end


function zstep(
    z::T,
    dz::T,
    grid::Grids.Grid,
    field::Fields.Field,
    guard::Guards.Guard,
    model::Model,
) where T<:AbstractFloat
    if model.PLASMA
        @timeit "plasma" begin
            solve!(field.rho, field.kdrho, field.E, grid.t, model.pe)
            CUDA.synchronize()
        end
    end

    if isa(grid, Grids.GridT) | isa(grid, Grids.GridRT) | isa(grid, Grids.GridXYT)
        @timeit "field -> spectr" begin
            forward_transform_time!(field.E, field.PT)
            CUDA.synchronize()
        end
    end

    if model.NONLINEARITY
        @timeit "nonlinearity" begin
           propagate!(field.E, model.NP, z, dz)
           CUDA.synchronize()
       end
    end

    @timeit "linear" begin
        propagate!(field.E, model.LP, dz)
        CUDA.synchronize()
    end

    if isa(grid, Grids.GridT) | isa(grid, Grids.GridRT) | isa(grid, Grids.GridXYT)
        @timeit "spectr -> field" begin
            inverse_transform_time!(field.E, field.PT)
            CUDA.synchronize()
        end
    end

    @timeit "field filter" begin
        Guards.apply_field_filter!(field.E, guard)
        CUDA.synchronize()
    end
    return nothing
end


# ******************************************************************************
function forward_transform_space!(E::AbstractArray, P::Nothing)
    return nothing
end


function inverse_transform_space!(E::AbstractArray, P::Nothing)
    return nothing
end


function forward_transform_space!(
    E::AbstractArray, P::Union{FFTW.Plan, HankelTransforms.Plan},
)
    P * E
    return nothing
end


function inverse_transform_space!(
    E::AbstractArray, P::Union{FFTW.Plan, HankelTransforms.Plan},
)
    P \ E
    return nothing
end


# ******************************************************************************
function forward_transform_time!(E::AbstractArray, P::Nothing)
    return nothing
end


function inverse_transform_time!(E::AbstractArray, P::Nothing)
    return nothing
end


function forward_transform_time!(E::AbstractArray, P::FFTW.Plan)
    P \ E   # time -> frequency [exp(-i*w*t)]
    return nothing
end


function inverse_transform_time!(E::AbstractArray, P::FFTW.Plan)
    P * E   # frequency -> time [exp(-i*w*t)]
    return nothing
end


# ******************************************************************************
function real_signal_to_analytic_spectrum!(E::AbstractArray, P::Nothing)
    return nothing
end


function real_signal_to_analytic_spectrum!(E::AbstractArray, P::FFTW.Plan)
    AnalyticSignals.rsig2aspec!(E, P)
    return nothing
end


end
